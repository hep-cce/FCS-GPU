/*
  Copyright (C) 2002-2017 CERN for the benefit of the ATLAS collaboration
*/


#include "ISF_FastCaloSimParametrization/FastCaloSimParamAlg.h"

  /**
   *
   *
   */

// STL include(s):
#include <sstream>
#include <map>

// local include(s):
//#include "LArG4Code/EnergySpot.h"
//#include "LArG4ShowerLib/Shower.h"
//#include "LArG4ShowerLib/ShowerLibList.h"

#include "ISF_FastCaloSimEvent/FCS_StepInfo.h"
#include "ISF_FastCaloSimEvent/FCS_StepInfoCollection.h"

// athena includes
#include "GeoModelInterfaces/IGeoModelSvc.h"
#include "EventInfo/TagInfo.h"


#include "CaloIdentifier/CaloCell_ID.h"
#include "CaloGeoHelpers/CaloSampling.h"
#include "CaloDetDescr/CaloDetDescrManager.h"
#include "CaloDetDescr/CaloDetDescrElement.h"


// For MC Truth information:
#include "GeneratorObjects/McEventCollection.h"

// geant includes
#include "G4Version.hh"
#include "TFile.h"


using CLHEP::Hep3Vector;

struct SortByE{
  bool operator() (const ISF_FCS_Parametrization::FCS_StepInfo& step1, const ISF_FCS_Parametrization::FCS_StepInfo& step2) { return (step1.energy() > step2.energy()); }
  bool operator() (const ISF_FCS_Parametrization::FCS_StepInfo* step1, const ISF_FCS_Parametrization::FCS_StepInfo* step2) { return (step1->energy() > step2->energy()); }
};

FastCaloSimParamAlg::FastCaloSimParamAlg(const std::string& name, ISvcLocator* pSvcLocator)
  : AthAlgorithm(name, pSvcLocator)
  , m_inputCollectionKey("EventSteps")
  , m_outputCollectionKey("MergedEventSteps")
  , m_calo_dd_man(nullptr)
{
  declareProperty("InputCollectionName", m_inputCollectionKey, "");
  declareProperty("OutputCollectionName", m_outputCollectionKey, "");
  declareProperty("Clusterize", m_clusterize = true, "merge nearby hits");
  declareProperty("Truncate", m_truncate = 2,"truncate hits with t>1000ns (if >=2)");
  declareProperty("MaxDistance",   m_maxDistance = 50000.,
                  "max distance squared after which the hits will be truncated");
  declareProperty("MinEnergy",   m_minEnergy = .99,
                  "energy border, that truncation won't cross");
  declareProperty("MaxRadius",        m_maxRadius = 25.,
                  "maximal radius squared until two hits will be combined");
  declareProperty("MaxRadiusLAr",        m_maxRadiusLAr = 25.,
                  "maximal radius in LAr squared until two hits will be combined");
  declareProperty("MaxRadiusHEC",        m_maxRadiusHEC = 100.,
                  "maximal radius in HEC squared until two hits will be combined");
  declareProperty("MaxRadiusFCAL",        m_maxRadiusFCAL = 100.,
                  "maximal radius in FCAL squared until two hits will be combined");
  declareProperty("MaxRadiusTile",        m_maxRadiusTile = 100.,
                  "maximal radius in Tile squared until two hits will be combined");

  declareProperty("MaxTime", m_maxTime = 25., "max time difference to merge two hits (ns) ");
  declareProperty("MaxTimeLAr", m_maxTimeLAr = 25., "max time difference to merge two hits (ns) ");
  declareProperty("MaxTimeHEC", m_maxTimeHEC = 25., "max time difference to merge two hits (ns) ");
  declareProperty("MaxTimeFCAL", m_maxTimeFCAL = 25., "max time difference to merge two hits (ns) ");
  declareProperty("MaxTimeTile", m_maxTimeTile = 25., "max time difference to merge two hits (ns) ");


  declareProperty("ContainmentEnergy",        m_containmentEnergy = 0.95,
                  "energy fraction that will be inside containment borders");
  declareProperty("LibStructFiles",   m_lib_struct_files,
                  "List of files to read library structures from");
  declareProperty("EnergyFraction",   m_energyFraction = .02,
                  "the allowed amount of energy that can be deposited outside calorimeter region ");
}


StatusCode FastCaloSimParamAlg::initialize()
{
  ATH_MSG_DEBUG("Initializing " << this->name() << " - package version " << PACKAGE_VERSION);
  ATH_CHECK(m_inputCollectionKey.initialize());
  ATH_CHECK(m_outputCollectionKey.initialize());
  m_calo_dd_man  = CaloDetDescrManager::instance();
  ATH_MSG_DEBUG("FastCaloSimParamAlg " << this->name() << " initialized");
  return StatusCode::SUCCESS;
}

StatusCode FastCaloSimParamAlg::execute()
{
  SG::ReadHandle<ISF_FCS_Parametrization::FCS_StepInfoCollection> inputHandle{m_inputCollectionKey};
  SG::WriteHandle<ISF_FCS_Parametrization::FCS_StepInfoCollection> outputHandle{m_outputCollectionKey};
  ATH_CHECK(outputHandle.record(std::make_unique<ISF_FCS_Parametrization::FCS_StepInfoCollection>()));

  // TODO would be more efficient to directly write the truncated
  // input collection to the output collection rather than copying it
  // first.
  for (const auto& step: *inputHandle ) {
    auto&& stepCopy = std::make_unique<ISF_FCS_Parametrization::FCS_StepInfo>(*step);
    outputHandle->push_back( stepCopy.release() );
  }
  ATH_CHECK(this->truncate(&*outputHandle));

  if (m_clusterize) {
    ATH_CHECK(this->clusterize(&*outputHandle));
  }
  else {
    ATH_MSG_DEBUG("Not merging nearby hits: "<<m_clusterize);
  }
  return StatusCode::SUCCESS;
}

StatusCode FastCaloSimParamAlg::clusterize(ISF_FCS_Parametrization::FCS_StepInfoCollection* stepinfo) const
{
  ATH_MSG_DEBUG("Initial clusterize size: "<<stepinfo->size()<<" - will merge steps in the same cell which are less than dR and dT to each other");
  double total_energy1(0.);
  for (const auto& step: *stepinfo) {
    total_energy1+=step->energy();
  }
  ATH_MSG_DEBUG("Check: total energy before clusterize "<<total_energy1);

  // Try this if it will be faster: split to cells first
  std::map<Identifier, ISF_FCS_Parametrization::FCS_StepInfoCollection*> FCS_SIC_cells;
  for (const auto& step: *stepinfo) {
    if (FCS_SIC_cells.find(step->identify()) != FCS_SIC_cells.end()) {// Already have a step for this cell
      auto&& stepCopy = std::make_unique<ISF_FCS_Parametrization::FCS_StepInfo>(*step);
      FCS_SIC_cells[step->identify()]->push_back( stepCopy.release() );
    }
    else { // First step for this cell
      auto && new_fcs_sic = std::make_unique<ISF_FCS_Parametrization::FCS_StepInfoCollection>();
      auto&& stepCopy = std::make_unique<ISF_FCS_Parametrization::FCS_StepInfo>(*step);
      new_fcs_sic->push_back( stepCopy.release() );
      FCS_SIC_cells.insert(std::pair<Identifier, ISF_FCS_Parametrization::FCS_StepInfoCollection*>(step->identify(),new_fcs_sic.release()));
    }
  }

  ATH_MSG_DEBUG("Merging separately in each cell: Ncells: "<<FCS_SIC_cells.size());
  // Then do merging for each cell
  for (std::map<Identifier, ISF_FCS_Parametrization::FCS_StepInfoCollection*>::iterator it = FCS_SIC_cells.begin(); it!= FCS_SIC_cells.end(); ++it) {
    std::stable_sort(FCS_SIC_cells[it->first]->begin(), FCS_SIC_cells[it->first]->end(), SortByE());
    if (! m_calo_dd_man->get_element(it->first)) {
        // Bad identifier
        ATH_MSG_WARNING("Something wrong with identifier: "<<it->first);
        continue;
      }
      else if ((it->second)->size()==1) {
        continue; // Go to next iterator
      }

    const CaloCell_ID::CaloSample layer = m_calo_dd_man->get_element(it->first)->getSampling();
    double dsame(0.);
    double tsame(0.);
    if (layer >= CaloCell_ID::PreSamplerB && layer <= CaloCell_ID::EME3) {
      dsame = m_maxRadiusLAr;
      tsame = m_maxTimeLAr;
    }
    else if (layer >= CaloCell_ID::HEC0  && layer <= CaloCell_ID::HEC3) {
      dsame = m_maxRadiusHEC;
      tsame = m_maxTimeHEC;
    }
    else if (layer >= CaloCell_ID::TileBar0 && layer <= CaloCell_ID::TileExt2) {
      dsame = m_maxRadiusTile;
      tsame = m_maxTimeTile;
    }
    else if (layer >=CaloCell_ID::FCAL0 && layer <= CaloCell_ID::FCAL2) {
      dsame = m_maxRadiusFCAL;
      tsame = m_maxTimeFCAL;
    }
    else {
      dsame = m_maxRadius;
      tsame = m_maxTime;
    }
    ISF_FCS_Parametrization::FCS_StepInfoCollection::iterator it1 = (it->second)->begin();
    while ( it1 != (it->second)->end() ) {
      ISF_FCS_Parametrization::FCS_StepInfoCollection::iterator it2 = it1; ++it2;
      while (it2 != (it->second)->end()) {
        if (((*it1)->diff2(**it2) < dsame) && std::fabs((*it1)->time() - (*it2)->time()) < tsame ) {
          **it1 += **it2;
          it2 = (it->second)->erase(it2); // also calls delete on the pointer (because the DataVector owns its elements)
          continue;
        }
        ++it2;
      }
      ++it1;
    }
  }
  // Merge them back into a single list
  ATH_MSG_VERBOSE("Copying back");
  stepinfo->clear(); // also calls delete on all the removed elements (because the DataVector owns its elements)
  for (std::map<Identifier, ISF_FCS_Parametrization::FCS_StepInfoCollection*>::iterator it = FCS_SIC_cells.begin(); it!= FCS_SIC_cells.end(); ++it) {
    for (const auto& step: *(it->second)) {
      auto&& stepCopy = std::make_unique<ISF_FCS_Parametrization::FCS_StepInfo>(*step);
      stepinfo->push_back( stepCopy.release() );
    }
    // Tidy up temporary FCS_StepInfoCollections as we go
    it->second->clear(); // also calls delete on all the removed elements (because the DataVector owns its elements)
    delete (it->second);
  }
  double total_energy2(0.);
  for (const auto& step: *stepinfo) {
    total_energy2+=step->energy();
  }
  ATH_MSG_DEBUG("Check: total energy "<<total_energy2);
  ATH_MSG_DEBUG("After clusterize: "<<stepinfo->size());
  unsigned int nInvalid(0);
  // Remove invalid FCS_StepInfo objects
  ISF_FCS_Parametrization::FCS_StepInfoCollection::iterator stepIter = stepinfo->begin();
  while(stepIter != stepinfo->end()) {
    if ((*stepIter)->valid()) {
      stepIter++;
      continue;
    }
    ++nInvalid;
    stepIter = stepinfo->erase(stepIter); // also calls delete on the pointer (because the DataVector owns its elements)
  }
  ATH_MSG_DEBUG("Removed "<<nInvalid<<" StepInfo objects. New collection size: "<<stepinfo->size());
  return StatusCode::SUCCESS;
}

StatusCode FastCaloSimParamAlg::truncate(ISF_FCS_Parametrization::FCS_StepInfoCollection* stepinfo) const
{
  ATH_MSG_DEBUG("Initial truncate size: "<<stepinfo->size()<<" settings: "<<m_truncate);
  if (m_truncate>0) {
    ISF_FCS_Parametrization::FCS_StepInfoCollection::iterator stepIter = stepinfo->begin();
    while (stepIter != stepinfo->end()) {
      if ((m_truncate>=2)&&((*stepIter)->time()>1000)) {
        stepIter = stepinfo->erase(stepIter); // also calls delete on the pointer (because the DataVector owns its elements)
        continue;
      }
      ++stepIter;
    }
    ATH_MSG_DEBUG("After truncate size: "<<stepinfo->size());
  }
  return StatusCode::SUCCESS;
}
