/*
  Copyright (C) 2002-2018 CERN for the benefit of the ATLAS collaboration
*/

#include "ISF_FastCaloSimEvent/TFCSLateralShapeParametrizationHitChain.h"
#include "ISF_FastCaloSimEvent/FastCaloSim_CaloCell_ID.h"

#include "ISF_FastCaloSimEvent/TFCSSimulationState.h"

//=============================================
//======= TFCSLateralShapeParametrization =========
//=============================================

TFCSLateralShapeParametrizationHitChain::TFCSLateralShapeParametrizationHitChain(const char* name, const char* title):TFCSLateralShapeParametrization(name,title),m_number_of_hits_simul(nullptr)
{
}

TFCSLateralShapeParametrizationHitChain::TFCSLateralShapeParametrizationHitChain(TFCSLateralShapeParametrizationHitBase* hitsim):TFCSLateralShapeParametrization(TString("hit_chain_")+hitsim->GetName(),TString("hit chain for ")+hitsim->GetTitle()),m_number_of_hits_simul(nullptr)
{
  set_pdgid_Ekin_eta_Ekin_bin_calosample(*hitsim);
  
  m_chain.push_back(hitsim);
}

void TFCSLateralShapeParametrizationHitChain::set_geometry(ICaloGeometry* geo)
{
  TFCSLateralShapeParametrization::set_geometry(geo);
  if(m_number_of_hits_simul) m_number_of_hits_simul->set_geometry(geo);
}

int TFCSLateralShapeParametrizationHitChain::get_number_of_hits(TFCSSimulationState& simulstate,const TFCSTruthState* truth, const TFCSExtrapolationState* extrapol) const
{
  // TODO: should we still do it?
  if(m_number_of_hits_simul) {
    int n=m_number_of_hits_simul->get_number_of_hits(simulstate,truth,extrapol);
    if(n<1) n=1;
    return n;
  }
  for(TFCSLateralShapeParametrizationHitBase* hitsim : m_chain) {
    int n=hitsim->get_number_of_hits(simulstate,truth,extrapol);
    if(n>0) return n;
  } 
  return 1;
}

FCSReturnCode TFCSLateralShapeParametrizationHitChain::simulate(TFCSSimulationState& simulstate,const TFCSTruthState* truth, const TFCSExtrapolationState* extrapol)
{
  // Call get_number_of_hits() only once, as it could contain a random number
  int nhit = get_number_of_hits(simulstate, truth, extrapol);
  if (nhit <= 0) {
    ATH_MSG_ERROR("TFCSLateralShapeParametrizationHitChain::simulate(): number of hits could not be calculated");
    return FCSFatal;
  }

  float Ehit=simulstate.E(calosample())/nhit;

  bool debug = msgLvl(MSG::DEBUG);
  if (debug) {
    ATH_MSG_DEBUG("E("<<calosample()<<")="<<simulstate.E(calosample())<<" #hits="<<nhit);
  }

  for (int i = 0; i < nhit; ++i) {
    TFCSLateralShapeParametrizationHitBase::Hit hit; 
    hit.E()=Ehit;
    for(TFCSLateralShapeParametrizationHitBase* hitsim : m_chain) {
      if (debug) {
        if (i < 2) hitsim->setLevel(MSG::DEBUG);
        else hitsim->setLevel(MSG::INFO);
      }

      for (int i = 0; i <= FCS_RETRY_COUNT; i++) {
        if (i > 0) ATH_MSG_WARNING("TFCSLateralShapeParametrizationHitChain::simulate(): Retry simulate_hit call " << i << "/" << FCS_RETRY_COUNT);
  
        FCSReturnCode status = hitsim->simulate_hit(hit, simulstate, truth, extrapol);

        if (status == FCSSuccess)
          break;
        else if (status == FCSFatal)
          return FCSFatal;

        if (i == FCS_RETRY_COUNT) {
          ATH_MSG_ERROR("TFCSLateralShapeParametrizationHitChain::simulate(): simulate_hit call failed after " << FCS_RETRY_COUNT << "retries");
        }
      }
    }
  }

  return FCSSuccess;
}

void TFCSLateralShapeParametrizationHitChain::Print(Option_t *option) const
{
  TFCSLateralShapeParametrization::Print(option);
  TString opt(option);
  bool shortprint=opt.Index("short")>=0;
  bool longprint=msgLvl(MSG::DEBUG) || (msgLvl(MSG::INFO) && !shortprint);
  TString optprint=opt;optprint.ReplaceAll("short","");

  if(m_number_of_hits_simul) {
    if(longprint) ATH_MSG_INFO(optprint <<"#:Number of hits simulation:");
    m_number_of_hits_simul->Print(opt+"#:");
  }
  if(longprint) ATH_MSG_INFO(optprint <<"- Simulation chain:");
  char count='A';
  for(TFCSLateralShapeParametrizationHitBase* hitsim : m_chain) {
    hitsim->Print(opt+count+" ");
    count++;
  } 
}
