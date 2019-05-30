#ifndef CALOGPUGENERAL_H
#define CALOGPUGENERAL_H

#include <iostream>
#include "Args.h"

namespace CaloGpuGeneral 
{
/*
// geometry
 void * cells_g ; // calo cells array 
 long long  ncells ; // total cells 
 void * regions_g ;
 unsigned int  nregions ; // number of regions
*/

void  GpuHitChain0 () ;

void  Gpu_Chain_Test();

void  simulate_hits( float, int, Chain0_Args  ) ;

}
#endif
