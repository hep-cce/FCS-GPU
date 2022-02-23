/*
  Copyright (C) 2002-2021 CERN for the benefit of the ATLAS collaboration
*/

#ifndef CALOGPUGENERAL_STDPAR_H
#define CALOGPUGENERAL_STDPAR_H

#include "Args.h"

namespace CaloGpuGeneral_stdpar {

  void simulate_hits_gr( Sim_Args& );
  void Rand4Hits_finish( void* );  
  void load_hitsim_params( void* rd4h, HitParams* hp, long* simbins, int bins );
  
} // namespace CaloGpuGeneral_stdpar
#endif
