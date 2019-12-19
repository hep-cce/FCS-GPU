#ifndef GPUARGS_H
#define GPUARGS_H

#include "GpuParams.h"
#include "FH_structs.h"
#include "GeoGpu_structs.h"
#include "Hit.h"

#include "GpuGeneral_structs.h"



typedef struct Sim_Args {

int debug ;
void * rd4h ;
GeoGpu * geo ;
float * cells_energy ; // big, all cells, ~ 200K array
Cell_E *  hitcells_E ; // array with only hit cells (Mem maxhitct )
Cell_E *  hitcells_E_h ; // array with only hit cells (Mem maxhitct )

int * ct ;
int * ct_h ;
HitParams * hitparams ;
long * simbins ; 
int nbins ;
int nsims ;
long nhits ;
long long ncells ;
float * rand ;

} Sim_Args ;

#endif
