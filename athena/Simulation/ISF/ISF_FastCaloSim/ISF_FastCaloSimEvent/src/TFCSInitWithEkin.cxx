/*
  Copyright (C) 2002-2018 CERN for the benefit of the ATLAS collaboration
*/

#include "ISF_FastCaloSimEvent/TFCSInitWithEkin.h"
#include "ISF_FastCaloSimEvent/TFCSSimulationState.h"
#include "ISF_FastCaloSimEvent/TFCSTruthState.h"

//=============================================
//======= TFCSInitWithEkin =========
//=============================================

TFCSInitWithEkin::TFCSInitWithEkin(const char* name, const char* title):TFCSParametrization(name,title)
{
  set_match_all_pdgid();
}

FCSReturnCode TFCSInitWithEkin::simulate(TFCSSimulationState& simulstate,const TFCSTruthState* truth, const TFCSExtrapolationState*)
{
  ATH_MSG_DEBUG("set E to Ekin="<<truth->Ekin());
  simulstate.set_E(truth->Ekin());
  return FCSSuccess;
}
