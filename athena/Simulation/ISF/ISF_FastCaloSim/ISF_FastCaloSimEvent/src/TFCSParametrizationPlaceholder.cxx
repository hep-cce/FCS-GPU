/*
  Copyright (C) 2002-2018 CERN for the benefit of the ATLAS collaboration
*/

#include "ISF_FastCaloSimEvent/TFCSParametrizationPlaceholder.h"

//=============================================
//======= TFCSParametrizationPlaceholder =========
//=============================================

FCSReturnCode TFCSParametrizationPlaceholder::simulate(TFCSSimulationState& /*simulstate*/,const TFCSTruthState* /*truth*/, const TFCSExtrapolationState* /*extrapol*/)
{
  ATH_MSG_ERROR("TFCSParametrizationPlaceholder::simulate(). This is a placeholder and should never get called. Likely a problem in the reading of the parametrization file occured and this class was not replaced with the real parametrization");
  return FCSFatal;
}

