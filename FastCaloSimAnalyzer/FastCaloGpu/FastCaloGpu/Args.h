/*
  Copyright (C) 2002-2021 CERN for the benefit of the ATLAS collaboration
*/

#ifndef GPUARGS_H
#define GPUARGS_H

#include "FH_structs.h"
#include "GpuGeneral_structs.h"
#include <chrono>

#ifdef USE_STDPAR
#include <atomic>
#endif

#define MAXHITS 200000
#define MAXBINS 1024
#define MAXHITCT 2000

class GeoGpu;

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
      count ++;
      return *this;
    }
    
    std::string print() const {
      std::string out;
      char buf[100];
      sprintf(buf,"%12s %15s %15s\n","kernel","total /s","avg launch /us");
      out += buf;
      sprintf(buf,"%12s %15.8f %15.1f\n","sim_clean",this->t_sim_clean.count(),
              this->t_sim_clean.count() * 1000000 /this->count);
      out += buf;
      sprintf(buf,"%12s %15.8f %15.1f\n","sim_A",this->t_sim_A.count(),
              this->t_sim_A.count() * 1000000 /this->count);
      out += buf;
      sprintf(buf,"%12s %15.8f %15.1f\n","sim_ct",this->t_sim_ct.count(),
              this->t_sim_ct.count() * 1000000 /this->count);
      out += buf;
      sprintf(buf,"%12s %15.8f %15.1f\n","sim_cp",this->t_sim_cp.count(),
              this->t_sim_cp.count() * 1000000 /this->count);
      out += buf;
      sprintf(buf,"%12s %15d\n","launch count",this->count);
      out += buf;
      
      return out;
    }
    
    friend std::ostream& operator<< (std::ostream& ost, const KernelTime& k) {
      return ost << k.print();
    }
    
  };
} // namespace CaloGpuGeneral

typedef struct Chain0_Args {

  bool debug;

  double extrapol_eta_ent;
  double extrapol_phi_ent;
  double extrapol_r_ent;
  double extrapol_z_ent;
  double extrapol_eta_ext;
  double extrapol_phi_ext;
  double extrapol_r_ext;
  double extrapol_z_ext;

  float extrapWeight;
  float E;

  int    pdgId;
  double charge;
  int    cs;
  bool   is_phi_symmetric;
  float* rand;
  int    nhits;
  void*  rd4h;

  FH2D* fh2d;
  FHs*  fhs;
  FH2D  fh2d_h; // host struct so we have info
  FHs   fhs_h;  // host struct

  GeoGpu* geo;

  bool is_first; // first event
  bool is_last;  // last event

  // bool * hitcells_b ;  // GPU array of whether a cell got hit
  // unsigned long * hitcells ;//GPU pointer for hit cell index for each hit
  // unsigned long * hitcells_l ; // GPU pointer for uniq  hitcell indexes
#ifdef USE_STDPAR
  std::atomic<int>* hitcells_ct;
#else
  int*          hitcells_ct; // GPU pointer for number of uniq hit cells
#endif
  unsigned long ncells;
  unsigned int  maxhitct;

  float*  cells_energy; // big, all cells, ~ 200K array
  Cell_E* hitcells_E;   // array with only hit cells (Mem maxhitct )
  Cell_E* hitcells_E_h; // host array

  unsigned int* hitcounts_b; // GPU pointer for interm blockwise result of hit counts

  // unsigned long * hitcells_h ; //Host array of hit cell index
  // int * hitcells_ct_h ; // host array of corresponding hit cell counts
  unsigned int ct; // cells got hit for the event

} Chain0_Args;

#endif
