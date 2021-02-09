/*
  Copyright (C) 2002-2021 CERN for the benefit of the ATLAS collaboration
*/

#ifndef CALOGPUGENERAL_H
#define CALOGPUGENERAL_H

#include <iostream>
#include "Args.h"

#define MIN_GPU_HITS 256

namespace CaloGpuGeneral {

  void* Rand4Hits_init( long long, unsigned short, unsigned long long, bool );
  void  Rand4Hits_finish( void* );

  void simulate_hits( float, int, Chain0_Args& );

} // namespace CaloGpuGeneral
#endif
