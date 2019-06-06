/*
  Copyright (C) 2002-2018 CERN for the benefit of the ATLAS collaboration
*/

#include "ISF_FastCaloSimEvent/TFCSHitCellMapping.h"
#include "ISF_FastCaloSimEvent/ICaloGeometry.h"
#include "ISF_FastCaloSimEvent/TFCSSimulationState.h"

//=============================================
//======= TFCSHitCellMapping =========
//=============================================

TFCSHitCellMapping::TFCSHitCellMapping(const char* name, const char* title, ICaloGeometry* geo) :
  TFCSLateralShapeParametrizationHitBase(name,title),
  m_geo(geo)
{
  set_match_all_pdgid();
}

FCSReturnCode TFCSHitCellMapping::simulate_hit(Hit& hit,TFCSSimulationState& simulstate,const TFCSTruthState* /*truth*/, const TFCSExtrapolationState* /*extrapol*/)
{
  int cs=calosample();
  const CaloDetDescrElement* cellele=m_geo->getDDE(cs,hit.eta(),hit.phi());
  ATH_MSG_DEBUG("HIT: cellele="<<cellele<<" E="<<hit.E()<<" cs="<<cs<<" eta="<<hit.eta()<<" phi="<<hit.phi());
  if(cellele) {
    simulstate.deposit(cellele,hit.E());
    return FCSSuccess;
  } else {
    ATH_MSG_ERROR("TFCSLateralShapeParametrizationHitCellMapping::simulate_hit: cellele="<<cellele<<" E="<<hit.E()<<" cs="<<cs<<" eta="<<hit.eta()<<" phi="<<hit.phi());
    return FCSFatal;
  }
}

void TFCSHitCellMapping::Print(Option_t *option) const
{
  TString opt(option);
  bool shortprint=opt.Index("short")>=0;
  bool longprint=msgLvl(MSG::DEBUG) || (msgLvl(MSG::INFO) && !shortprint);
  TString optprint=opt;optprint.ReplaceAll("short","");
  TFCSLateralShapeParametrizationHitBase::Print(option);

  if(longprint) ATH_MSG_INFO(optprint <<"  geo="<<m_geo);
}
