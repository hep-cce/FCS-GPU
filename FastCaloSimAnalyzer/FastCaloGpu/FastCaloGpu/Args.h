/*
  Copyright (C) 2002-2021 CERN for the benefit of the ATLAS collaboration
*/

#ifndef GPUARGS_H
#define GPUARGS_H

#include "FH_structs.h"
#include "GpuGeneral_structs.h"
#include <chrono>
#include <vector>
#include <cmath>

#ifdef USE_STDPAR
#include <atomic>
#endif

#define MAXHITS 200000
#define MAXBINS 1024
#define MAXHITCT 2000

class GeoGpu;

namespace CaloGpuGeneral {
  struct KernelTime {
    std::vector<std::chrono::duration<double> > t_sim_clean{ 0 };
    std::vector<std::chrono::duration<double> > t_sim_A{ 0 };
    std::vector<std::chrono::duration<double> > t_sim_ct{ 0 };
    std::vector<std::chrono::duration<double> > t_sim_cp{ 0 };
    unsigned int count{ 0 };
    KernelTime() = default;
    void add(std::chrono::duration<double> t1, std::chrono::duration<double> t2,
             std::chrono::duration<double> t3, std::chrono::duration<double> t4) {
      t_sim_clean.push_back(t1);
      t_sim_A.push_back(t2);
      t_sim_ct.push_back(t3);
      t_sim_cp.push_back(t4);
      count++;
    }
    
    void printAll() const {
      std::cout << "All kernel timings [" << t_sim_clean.size() << "]:\n";
      for (size_t i = 0; i < t_sim_clean.size(); ++i) {
        printf("%15.1f %15.1f %15.1f %15.1f\n", t_sim_clean[i].count() * 1000000,
               t_sim_A[i].count() * 1000000, t_sim_ct[i].count() * 1000000,
               t_sim_cp[i].count() * 1000000);
      }
    }
    
    std::string print() const {
      // Ignore first and last timing entries

      if (count <= 2) {
        return ("kernel timing: insufficient data\n");
      }      
      
      double s_cl{ 0. }, s_A{ 0. }, s_ct{ 0. }, s_cp{ 0. };
      for (size_t i = 1; i < t_sim_clean.size() - 1; ++i) {
        s_cl += t_sim_clean[i].count();
        s_A  += t_sim_A[i].count();
        s_ct += t_sim_ct[i].count();
        s_cp += t_sim_cp[i].count();
      }
      
      double ss_cl{ 0. }, ss_A{ 0. }, ss_ct{ 0. }, ss_cp{ 0. };
      for (size_t i = 1; i < t_sim_clean.size() - 1; ++i) {
        ss_cl += std::pow(t_sim_clean[i].count() - s_cl / (count - 2), 2);
        ss_A  += std::pow(t_sim_A[i].count() - s_A / (count - 2), 2);
        ss_ct += std::pow(t_sim_ct[i].count() - s_ct / (count - 2), 2);
        ss_cp += std::pow(t_sim_cp[i].count() - s_cp / (count - 2), 2);
      }
      
      ss_cl = 1000000 * std::sqrt(ss_cl / (count - 2));
      ss_A  = 1000000 * std::sqrt(ss_A / (count - 2));
      ss_ct = 1000000 * std::sqrt(ss_ct / (count - 2));
      ss_cp = 1000000 * std::sqrt(ss_cp / (count - 2));
      
      std::string out("kernel timing\n");
      char buf[100];
      sprintf(buf, "%12s %15s %15s %15s\n", "kernel", "total /s",
              "avg launch /us", "std dev /us");
      out += buf;
      sprintf(buf, "%12s %15.8f %15.1f %15.1f\n", "sim_clean", s_cl,
              s_cl * 1000000 / (count - 2), ss_cl);
      out += buf;
      sprintf(buf, "%12s %15.8f %15.1f %15.1f\n", "sim_A", s_A,
              s_A * 1000000 / (count - 2), ss_A);
      out += buf;
      sprintf(buf, "%12s %15.8f %15.1f %15.1f\n", "sim_ct", s_ct,
              s_ct * 1000000 / (count - 2), ss_ct);
      out += buf;
      sprintf(buf, "%12s %15.8f %15.1f %15.1f\n", "sim_cp", s_cp,
              s_cp * 1000000 / (count - 2), ss_cp);
      out += buf;
      sprintf(buf, "%12s %15d +2\n", "launch count", count - 2);
      out += buf;
      
      return out;
    }
    
    friend std::ostream &operator<<(std::ostream &ost, const KernelTime &k) {
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

  // bool *          hitcells_b ;  // GPU array of whether a cell got hit
  // unsigned long * hitcells ;//GPU pointer for hit cell index for each hit
  // unsigned long * hitcells_l ; // GPU pointer for uniq  hitcell indexes

  CELL_CT_T*         hitcells_ct; // GPU pointer for number of uniq hit cells
  CELL_ENE_T*        cells_energy; // big, all cells, ~ 200K array

  unsigned long      ncells;
  unsigned int       maxhitct;

  Cell_E*            hitcells_E;   // array with only hit cells (Mem maxhitct )
  Cell_E*            hitcells_E_h; // host array

  unsigned int*      hitcounts_b; // GPU pointer for interm blockwise result of hit counts

  // unsigned long * hitcells_h ; //Host array of hit cell index
  // int *           hitcells_ct_h ; // host array of corresponding hit cell counts
  unsigned int       ct; // cells got hit for the event

} Chain0_Args;

#endif
