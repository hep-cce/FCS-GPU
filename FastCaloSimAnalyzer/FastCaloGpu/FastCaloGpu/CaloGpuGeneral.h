#ifndef CALOGPUGENERAL_H
#define CALOGPUGENERAL_H

#include <iostream>
#include "Args.h"

#define  MIN_GPU_HITS 1000 

namespace CaloGpuGeneral 
{

void  GpuHitChain0 () ;

void  Gpu_Chain_Test();

void *   Rand4Hits_init(long long ,unsigned short,  unsigned long long);
void    Rand4Hits_finish(void *);

void  simulate_hits( float, int, Chain0_Args&  ) ;

}
#endif
