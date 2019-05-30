/*
  Copyright (C) 2002-2017 CERN for the benefit of the ATLAS collaboration
*/

#include "ISF_FastCaloSimEvent/TFCS2DFunction.h"

//=============================================
//======= TFCS2DFunction =========
//=============================================

void TFCS2DFunction::rnd_to_fct(float value[],const float rnd[]) const
{
  rnd_to_fct(value[0],value[1],rnd[0],rnd[1]);
}
