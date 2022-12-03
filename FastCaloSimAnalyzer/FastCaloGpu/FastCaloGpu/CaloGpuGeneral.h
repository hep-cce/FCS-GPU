/*
  Copyright (C) 2002-2023 CERN for the benefit of the ATLAS collaboration
*/

#ifndef CALOGPUGENERAL_H
#define CALOGPUGENERAL_H

#include <iostream>
#include "Args.h"


namespace CaloGpuGeneral 
{



void *   Rand4Hits_init(long long ,int,  unsigned long long ,bool );
void    Rand4Hits_finish(void *);

void  load_hitsim_params( void *, HitParams *, long * , int ) ;


void  simulate_hits_gr( Sim_Args&  ) ;

}
#endif
