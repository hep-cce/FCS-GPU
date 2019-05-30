/*
  Copyright (C) 2002-2018 CERN for the benefit of the ATLAS collaboration
*/

#include "ISF_FastCaloSimEvent/TFCSHitCellMappingFCal.h"
#include "ISF_FastCaloSimEvent/ICaloGeometry.h"
#include "ISF_FastCaloSimEvent/TFCSSimulationState.h"

//=============================================
//======= TFCSHitCellMappingFCal =========
//=============================================


FCSReturnCode TFCSHitCellMappingFCal::simulate_hit(Hit& hit,TFCSSimulationState& simulstate,const TFCSTruthState* /*truth*/, const TFCSExtrapolationState* /*extrapol*/)
{
  int cs=calosample();
  const CaloDetDescrElement* cellele=m_geo->getFCalDDE(cs,hit.x(),hit.y(),hit.z());
  ATH_MSG_DEBUG("HIT: cellele="<<cellele<<" E="<<hit.E()<<" cs="<<cs<<" x="<<hit.x()<<" y="<<hit.y() << " z="<<hit.z());
  if(cellele) {
    simulstate.deposit(cellele,hit.E());
  } else {
    ATH_MSG_ERROR("TFCSLateralShapeParametrizationHitCellMapping::simulate_hit: cellele="<<cellele<<" E="<<hit.E()<<" cs="<<cs<<" eta="<<hit.eta()<<" phi="<<hit.phi());
    return FCSFatal;
  }

  return FCSSuccess;
}
