/*
  Copyright (C) 2002-2018 CERN for the benefit of the ATLAS collaboration
*/

#include "ISF_FastCaloSimEvent/TFCSCenterPositionCalculation.h"
#include "ISF_FastCaloSimEvent/FastCaloSim_CaloCell_ID.h"

#include "ISF_FastCaloSimEvent/TFCSSimulationState.h"
#include "ISF_FastCaloSimEvent/TFCSExtrapolationState.h"

//=============================================
//======= TFCSCenterPositionCalculation =========
//=============================================

TFCSCenterPositionCalculation::TFCSCenterPositionCalculation(const char* name, const char* title):TFCSLateralShapeParametrizationHitBase(name,title),m_extrapWeight(0.5)
{
}

FCSReturnCode TFCSCenterPositionCalculation::simulate_hit(Hit& hit,TFCSSimulationState& /*simulstate*/,const TFCSTruthState* /*truth*/, const TFCSExtrapolationState* extrapol)
{
   const int cs=calosample();
   hit.setCenter_r( (1.-m_extrapWeight)*extrapol->r(cs, SUBPOS_ENT) + m_extrapWeight*extrapol->r(cs, SUBPOS_EXT) );
   hit.setCenter_z( (1.-m_extrapWeight)*extrapol->z(cs, SUBPOS_ENT) + m_extrapWeight*extrapol->z(cs, SUBPOS_EXT) );
   hit.setCenter_eta( (1.-m_extrapWeight)*extrapol->eta(cs, SUBPOS_ENT) + m_extrapWeight*extrapol->eta(cs, SUBPOS_EXT) );
   hit.setCenter_phi( (1.-m_extrapWeight)*extrapol->phi(cs, SUBPOS_ENT) + m_extrapWeight*extrapol->phi(cs, SUBPOS_EXT) );
   
   ATH_MSG_DEBUG("TFCSCenterPositionCalculation: center_r: " << hit.center_r() << " center_z: " << hit.center_z() << " center_phi: " << hit.center_phi() << " center_eta: " << hit.center_eta() << " weight: " << m_extrapWeight << " cs: " << cs);
   
   return FCSSuccess;
}

void TFCSCenterPositionCalculation::Print(Option_t* option) const
{   
   TString opt(option);
   bool shortprint=opt.Index("short")>=0;
   bool longprint=msgLvl(MSG::DEBUG) || (msgLvl(MSG::INFO) && !shortprint);
   TString optprint=opt;optprint.ReplaceAll("short","");
   TFCSLateralShapeParametrizationHitBase::Print(option);
   
   if(longprint) ATH_MSG_INFO(optprint << "  Weight for extrapolated position: " << m_extrapWeight);
}
