/*
  Copyright (C) 2002-2021 CERN for the benefit of the ATLAS collaboration
*/

#ifndef CALOGPUGENERAL_kk_H
#define CALOGPUGENERAL_kk_H

#include "Args.h"

namespace CaloGpuGeneral_kk {

  void simulate_hits_gr( Sim_Args& );
  void Rand4Hits_finish( void* );

  void load_hitsim_params( void*, HitParams*, long*, int );

} // namespace CaloGpuGeneral_kk
#endif
