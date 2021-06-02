#ifndef GPUARGS_H
#define GPUARGS_H

#include "GpuParams.h"
#include "FH_structs.h"
#include "GeoGpu_structs.h"
#include "Hit.h"
#include <chrono>

#include "GpuGeneral_structs.h"


namespace CaloGpuGeneral {
    struct KernelTime {
      std::chrono::duration<double> t_sim_clean {0};
      std::chrono::duration<double> t_sim_A {0};
      std::chrono::duration<double> t_sim_ct {0};
      std::chrono::duration<double> t_sim_cp {0};
      KernelTime() = default;
      KernelTime( std::chrono::duration<double> t1, std::chrono::duration<double> t2,
                  std::chrono::duration<double> t3, std::chrono::duration<double> t4 ):
        t_sim_clean(t1),t_sim_A(t2),t_sim_ct(t3),t_sim_cp(t4) {}
      KernelTime& operator+=(const KernelTime& rhs) {
        t_sim_clean += rhs.t_sim_clean;
        t_sim_A += rhs.t_sim_A;
        t_sim_ct += rhs.t_sim_ct;
        t_sim_cp += rhs.t_sim_cp;
        return *this;
      }
  };
}

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
