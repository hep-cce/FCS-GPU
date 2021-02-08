/*
  Copyright (C) 2002-2018 CERN for the benefit of the ATLAS collaboration
*/

#include "CLHEP/Random/RandPoisson.h"

#include "ISF_FastCaloSimEvent/TFCSLateralShapeParametrizationHitNumberFromE.h"
#include "ISF_FastCaloSimEvent/TFCSSimulationState.h"

#include "TMath.h"

//=============================================
//======= TFCSHistoLateralShapeParametrization =========
//=============================================

TFCSLateralShapeParametrizationHitNumberFromE::TFCSLateralShapeParametrizationHitNumberFromE(const char* name, const char* title,double stochastic,double constant) :
  TFCSLateralShapeParametrizationHitBase(name,title),m_stochastic(stochastic),m_constant(constant)
{
  set_match_all_pdgid();
}

int TFCSLateralShapeParametrizationHitNumberFromE::get_number_of_hits(TFCSSimulationState& simulstate,const TFCSTruthState* /*truth*/, const TFCSExtrapolationState* /*extrapol*/) const
{
  if (!simulstate.randomEngine()) {
    return -1;
  }

  int cs=calosample();
  double energy=simulstate.E(cs);

  if (energy < 0) {
    return 1;
  }

  if (TMath::IsNaN(energy)) {
    ATH_MSG_DEBUG("Energy is NaN");
    return 1;
  }
  
  double sigma_stochastic=m_stochastic/sqrt(energy/1000.0);
  int hits = CLHEP::RandPoisson::shoot(simulstate.randomEngine(), 1.0 / (sigma_stochastic*sigma_stochastic + m_constant*m_constant));

  ATH_MSG_DEBUG("#hits="<<hits);
  
  return hits;
}

void TFCSLateralShapeParametrizationHitNumberFromE::Print(Option_t *option) const
{
  TString opt(option);
  bool shortprint=opt.Index("short")>=0;
  bool longprint=msgLvl(MSG::DEBUG) || (msgLvl(MSG::INFO) && !shortprint);
  TString optprint=opt;optprint.ReplaceAll("short","");
  TFCSLateralShapeParametrizationHitBase::Print(option);

  if(longprint) ATH_MSG_INFO(optprint <<"  stochastic="<<m_stochastic<<" constant="<<m_constant);
}
