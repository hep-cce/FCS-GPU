#ifndef CALOGPUGENERAL_H
#define CALOGPUGENERAL_H

#include "Args.h"
#include <iostream>

#define MIN_GPU_HITS 256

namespace CaloGpuGeneral {

  void GpuHitChain0();

  void Gpu_Chain_Test();

  void* Rand4Hits_init( long long, unsigned short, unsigned long long, bool );
  void  Rand4Hits_finish( void* );

  void simulate_hits( float, int, Chain0_Args& );

} // namespace CaloGpuGeneral
#endif
