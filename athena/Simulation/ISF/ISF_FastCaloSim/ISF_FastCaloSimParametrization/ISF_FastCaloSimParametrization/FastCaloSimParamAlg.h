/*
  Copyright (C) 2002-2017 CERN for the benefit of the ATLAS collaboration
*/


#ifndef ISF_FASTCALOSIMPARAMETRIZATION_ISF_FastCaloSimParamAlg_H
#define ISF_FASTCALOSIMPARAMETRIZATION_ISF_FastCaloSimParamAlg_H


// STL includes
#include <string>
#include <list>
#include <map>

#include "AthenaBaseComps/AthAlgorithm.h"

#include "StoreGate/ReadHandleKey.h"
#include "StoreGate/WriteHandleKey.h"

#include "CaloDetDescr/CaloDetDescrManager.h"
//#include "CaloIdentifier/CaloCell_ID.h"
// CLHEP include(s)
#include "CLHEP/Vector/ThreeVector.h"

// forward includes in namespaces

namespace ISF_FCS_Parametrization {
  class FCS_StepInfoCollection;
  class FCS_StepInfo;
  typedef std::list<FCS_StepInfo*> FCS_StepInfoList;
}
namespace HepMC {
  class GenParticle;
}

  /**
   *
   *   @short Modified class for shower library generation algorithm
   *
   */
class FastCaloSimParamAlg : public AthAlgorithm {

 public:

  FastCaloSimParamAlg(const std::string& name, ISvcLocator* pSvcLocator);

  virtual StatusCode initialize() override final;
  virtual StatusCode execute() override final;

 private:

  StatusCode clusterize(ISF_FCS_Parametrization::FCS_StepInfoCollection* stepinfo) const;

  // const ISF_FCS_Parametrization::FCS_StepInfoCollection* getFCS_StepInfo();

  // ISF_FCS_Parametrization::FCS_StepInfoCollection* copyFCS_StepInfo(const ISF_FCS_Parametrization::FCS_StepInfoCollection* stepinfo);

  //ISF_FCS_Parametrization::FCS_StepInfoList* copyFCS_StepInfoZeroCleanup(const ISF_FCS_Parametrization::FCS_StepInfoCollection* stepinfo);

  StatusCode truncate(ISF_FCS_Parametrization::FCS_StepInfoCollection* stepinfo) const;

  /* data members */
  SG::ReadHandleKey<ISF_FCS_Parametrization::FCS_StepInfoCollection> m_inputCollectionKey;
  SG::WriteHandleKey<ISF_FCS_Parametrization::FCS_StepInfoCollection> m_outputCollectionKey;
  BooleanProperty           m_clusterize; //Y/N to merge nearby hits
  DoubleProperty            m_truncate; // Drop E<0 (if >1), t>1000 (if >2)
  DoubleProperty            m_maxDistance;
  DoubleProperty            m_maxRadius;          //!< property, see @link LArG4GenShowerLib::LArG4GenShowerLib @endlink
  DoubleProperty            m_maxRadiusLAr;            //!< property, see @link LArG4GenShowerLib::LArG4GenShowerLib @endlink
  DoubleProperty            m_maxRadiusHEC;            //!< property, see @link LArG4GenShowerLib::LArG4GenShowerLib @endlink
  DoubleProperty            m_maxRadiusFCAL;            //!< property, see @link LArG4GenShowerLib::LArG4GenShowerLib @endlink
  DoubleProperty            m_maxRadiusTile;            //!< property, see @link LArG4GenShowerLib::LArG4GenShowerLib @endlink

  DoubleProperty            m_maxTime;
  DoubleProperty            m_maxTimeLAr;
  DoubleProperty            m_maxTimeHEC;
  DoubleProperty            m_maxTimeFCAL;
  DoubleProperty            m_maxTimeTile;


  DoubleProperty            m_minEnergy;            //!< property, see @link LArG4GenShowerLib::LArG4GenShowerLib @endlink
  DoubleProperty            m_containmentEnergy;    //!< property, see @link LArG4GenShowerLib::LArG4GenShowerLib @endlink
  DoubleProperty            m_energyFraction;       //!< property, see @link LArG4GenShowerLib::LArG4GenShowerLib @endlink

  StringArrayProperty       m_lib_struct_files;

  const CaloDetDescrManager* m_calo_dd_man;

}; //

#endif // ISF_FASTCALOSIMPARAMETRIZATION_ISF_FastCaloSimParamAlg_H
