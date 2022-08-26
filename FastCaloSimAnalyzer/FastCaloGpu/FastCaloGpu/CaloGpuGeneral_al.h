/*
  Copyright (C) 2002-2021 CERN for the benefit of the ATLAS collaboration
*/

#ifndef CALOGPUGENERAL_AL_H
#define CALOGPUGENERAL_AL_H

#include "Args.h"

namespace CaloGpuGeneral_al {

  void Rand4Hits_finish( void* rd4h );
  void simulate_hits( float E, int nhits, Chain0_Args& args );


  void simulate_clean_alpaka(Chain0_Args& args);
  void simulate_A_alpaka( float E, int nhits, Chain0_Args& args);
  void simulate_ct_alpaka(Chain0_Args& args);

} // namespace CaloGpuGeneral_al
#endif
