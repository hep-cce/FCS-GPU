/*
  Copyright (C) 2002-2021 CERN for the benefit of the ATLAS collaboration
*/

#ifndef GPUARGS_H
#define GPUARGS_H

#include "GpuParams.h"
#include "FH_structs.h"
#include "GeoGpu_structs.h"
#include "Hit.h"
#include <chrono>

#ifdef USE_STDPAR
#  include <atomic>
#endif

#include "GpuGeneral_structs.h"

namespace CaloGpuGeneral {
  struct KernelTime {
    std::chrono::duration<double> t_sim_clean{0};
    std::chrono::duration<double> t_sim_A{0};
    std::chrono::duration<double> t_sim_ct{0};
    std::chrono::duration<double> t_sim_cp{0};
    unsigned int                  count{0};
    KernelTime() = default;
    KernelTime( std::chrono::duration<double> t1, std::chrono::duration<double> t2, std::chrono::duration<double> t3,
                std::chrono::duration<double> t4 )
        : t_sim_clean( t1 ), t_sim_A( t2 ), t_sim_ct( t3 ), t_sim_cp( t4 ) {}
    KernelTime& operator+=( const KernelTime& rhs ) {
      t_sim_clean += rhs.t_sim_clean;
      t_sim_A += rhs.t_sim_A;
      t_sim_ct += rhs.t_sim_ct;
      t_sim_cp += rhs.t_sim_cp;
      count++;
      return *this;
    }
  };
} // namespace CaloGpuGeneral

typedef struct Sim_Args {

  int count{0};

  int     debug{0};
  void*   rd4h{nullptr};
  GeoGpu* geo{nullptr};

  CELL_ENE_T* cells_energy{nullptr}; // big, all cells, ~ 200K array
  Cell_E*     hitcells_E{nullptr};   // array with only hit cells (Mem maxhitct )
  Cell_E*     hitcells_E_h{nullptr}; // array with only hit cells (Mem maxhitct )

#ifdef USE_STDPAR
  std::atomic<int>* ct{nullptr};
#else
  int* ct{nullptr}; // GPU pointer for number of uniq hit cells
#endif
  int* ct_h{nullptr};

  HitParams*    hitparams{nullptr};
  long*         simbins{nullptr};
  int           nbins{0};
  int           nsims{0};
  long          nhits{0};
  unsigned long ncells{0};
  unsigned int  maxhitct{0};
  float*        rand{nullptr};

} Sim_Args;

#endif
