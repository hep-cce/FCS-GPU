/*
  Copyright (C) 2002-2018 CERN for the benefit of the ATLAS collaboration
*/

#include "CLHEP/Random/RandFlat.h"

#include "ISF_FastCaloSimEvent/TFCSHitCellMappingWiggle.h"
#include "ISF_FastCaloSimEvent/TFCSSimulationState.h"
#include "ISF_FastCaloSimEvent/TFCSTruthState.h"
#include "ISF_FastCaloSimEvent/TFCSExtrapolationState.h"
#include "ISF_FastCaloSimEvent/TFCS1DFunctionInt32Histogram.h"

#include "TH1.h"
#include "TVector2.h"
#include "TMath.h"

#if 0 //defined(__FastCaloSimStandAlone__)
#include "CaloGeometryFromFile.h"
#endif

//=============================================
//======= TFCSHitCellMappingWiggle =========
//=============================================

TFCSHitCellMappingWiggle::TFCSHitCellMappingWiggle(const char* name, const char* title, ICaloGeometry* geo) : TFCSHitCellMapping(name,title,geo)
{
}

TFCSHitCellMappingWiggle::~TFCSHitCellMappingWiggle()
{
  for(auto function : m_functions) delete function;
}

void TFCSHitCellMappingWiggle::initialize(TFCS1DFunction* func)
{
  if(!func) return;
  for(auto function : m_functions) if(function) delete function;

  m_functions.resize(1);
  m_functions[0]=func;

  m_bin_low_edge.resize(2);
  m_bin_low_edge[0] = 0;
  m_bin_low_edge[1] = init_eta_max;
}

void TFCSHitCellMappingWiggle::initialize(const std::vector< const TFCS1DFunction* >& functions, const std::vector< float >& bin_low_edges)
{
  if(functions.size()+1!=bin_low_edges.size()) {
    ATH_MSG_ERROR("Using "<<functions.size()<<" functions needs "<<functions.size()+1<<" bins, but got "<<bin_low_edges.size()<<"bins");
    return;
  }
  for(auto function : m_functions) if(function) delete function;
  m_functions=functions;
  m_bin_low_edge=bin_low_edges;
}

void TFCSHitCellMappingWiggle::initialize(TH1* histogram,float xscale)
{
  if(!histogram) return;
  TFCS1DFunctionInt32Histogram* func=new TFCS1DFunctionInt32Histogram(histogram);
  if(xscale!=1) {
    for(auto& ele : func->get_HistoBordersx()) ele*=xscale;
  }
  initialize(func);
}

void TFCSHitCellMappingWiggle::initialize(const std::vector< const TH1* > histograms, std::vector< float > bin_low_edges, float xscale)
{
  if(histograms.size()+1!=bin_low_edges.size()) {
    ATH_MSG_ERROR("Using "<<histograms.size()<<" histograms needs "<<histograms.size()+1<<" bins, but got "<<bin_low_edges.size()<<"bins");
    return;
  }
  std::vector< const TFCS1DFunction* > functions(histograms.size());
  for(unsigned int i=0;i<histograms.size();++i) {
    if(histograms[i]) {
      TFCS1DFunctionInt32Histogram* func=new TFCS1DFunctionInt32Histogram(histograms[i]);
      if(xscale!=1) {
        for(auto& ele : func->get_HistoBordersx()) ele*=xscale;
      }
      functions[i]=func;
    } else {
      functions[i]=nullptr;
    }  
  }
  
  initialize(functions,bin_low_edges);
}

FCSReturnCode TFCSHitCellMappingWiggle::simulate_hit(Hit& hit,TFCSSimulationState& simulstate,const TFCSTruthState* truth, const TFCSExtrapolationState* extrapol)
{
  if (!simulstate.randomEngine()) {
    return FCSFatal;
  }

  float eta=fabs(hit.eta());
  if(eta<m_bin_low_edge[0] || eta>=m_bin_low_edge[get_number_of_bins()]) {
    return TFCSHitCellMapping::simulate_hit(hit,simulstate,truth,extrapol);
  }  
  
  auto it = std::upper_bound(m_bin_low_edge.begin(),m_bin_low_edge.end(),eta);
  int bin=std::distance(m_bin_low_edge.begin(),it)-1;

  const TFCS1DFunction* func=get_function(bin);
  if(func) {
    double rnd = CLHEP::RandFlat::shoot(simulstate.randomEngine());

    double wiggle=func->rnd_to_fct(rnd);

    ATH_MSG_DEBUG("HIT: E="<<hit.E()<<" cs="<<calosample()<<" eta="<<hit.eta()<<" phi="<<hit.phi()<<" wiggle="<<wiggle<<" bin="<<bin<<" ["<<get_bin_low_edge(bin)<<","<<get_bin_up_edge(bin)<<"] func="<<func);

    double hit_phi_shifted=hit.phi()+wiggle;
    hit.phi()=TVector2::Phi_mpi_pi(hit_phi_shifted);
  }  

  return TFCSHitCellMapping::simulate_hit(hit,simulstate,truth,extrapol);
}

void TFCSHitCellMappingWiggle::Print(Option_t *option) const
{
  TFCSHitCellMapping::Print(option);
  TString opt(option);
  bool shortprint=opt.Index("short")>=0;
  bool longprint=msgLvl(MSG::DEBUG) || (msgLvl(MSG::INFO) && !shortprint);
  TString optprint=opt;optprint.ReplaceAll("short","");
  
  if(longprint) {
    ATH_MSG(INFO) << optprint <<"  "<<get_number_of_bins()<<" functions in [";
    for (unsigned int i=0;i<get_number_of_bins();++i) msg()<<get_bin_low_edge(i)<<", ";
    msg()<<get_bin_up_edge(get_number_of_bins()-1)<<"]"<< endmsg;
  }  
}

void TFCSHitCellMappingWiggle::unit_test(TFCSSimulationState* simulstate,TFCSTruthState* truth, TFCSExtrapolationState* extrapol)
{
  if(!simulstate) simulstate=new TFCSSimulationState();
  if(!truth) truth=new TFCSTruthState();
  if(!extrapol) extrapol=new TFCSExtrapolationState();
  
  int nbin=10;
  float maxeta=5.0;
  std::vector< const TFCS1DFunction* > functions;
  std::vector< float > bin_low_edges;
  
  TFCSHitCellMappingWiggle wiggle_test("WiggleTest","WiggleTest");
  
  for(float eta=0;eta<maxeta;eta+=maxeta/nbin) {
    TH1* hist=TFCS1DFunction::generate_histogram_random_gauss(16,100000,-0.0125,0.0125,0,0.005);
    bin_low_edges.push_back(eta);
    functions.push_back(new TFCS1DFunctionInt32Histogram(hist));
    delete hist;
  }  
  bin_low_edges.push_back(100);
  wiggle_test.initialize(functions,bin_low_edges);
  wiggle_test.set_calosample(2);
  wiggle_test.setLevel(MSG::DEBUG);
  wiggle_test.Print();

#if 0 // defined(__FastCaloSimStandAlone__)
  CaloGeometryFromFile* geo = new CaloGeometryFromFile();

// * load geometry files
  geo->LoadGeometryFromFile("/afs/cern.ch/atlas/groups/Simulation/FastCaloSimV2/Geometry-ATLAS-R2-2016-01-00-01.root", "ATLAS-R2-2016-01-00-01");
  TString path_to_fcal_geo_files = "/afs/cern.ch/atlas/groups/Simulation/FastCaloSimV2/";
  geo->LoadFCalGeometryFromFiles(path_to_fcal_geo_files + "FCal1-electrodes.sorted.HV.09Nov2007.dat", path_to_fcal_geo_files + "FCal2-electrodes.sorted.HV.April2011.dat", path_to_fcal_geo_files + "FCal3-electrodes.sorted.HV.09Nov2007.dat");

  wiggle_test.set_geometry(geo);
  for(float eta=-maxeta+0.01;eta<maxeta;eta+=maxeta/nbin) {
    Hit hit;
    hit.eta()=eta;
    hit.phi()=0;
    hit.E()=1;
    wiggle_test.simulate_hit(hit,*simulstate,truth,extrapol);
  }
#endif

}
