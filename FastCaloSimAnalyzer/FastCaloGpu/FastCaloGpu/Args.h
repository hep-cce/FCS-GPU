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
#include <vector>

#ifdef USE_STDPAR
#  include <atomic>
#endif

#include "GpuGeneral_structs.h"

namespace CaloGpuGeneral {
  struct KernelTime {
    std::vector<std::chrono::duration<double>> t_sim_clean{0};
    std::vector<std::chrono::duration<double>> t_sim_A{0};
    std::vector<std::chrono::duration<double>> t_sim_ct{0};
    std::vector<std::chrono::duration<double>> t_sim_cp{0};
    unsigned int                  count{0};
    KernelTime() = default;
    void add( std::chrono::duration<double> t1, std::chrono::duration<double> t2, std::chrono::duration<double> t3,
              std::chrono::duration<double> t4 ) {
      t_sim_clean.push_back(t1);
      t_sim_A.push_back(t2);
      t_sim_ct.push_back(t3);
      t_sim_cp.push_back(t4);
      count++;
    }

    void printAll() const {
      std::cout << "All kernel timings [" << t_sim_clean.size() << "]:\n";
      for (size_t i=0; i<t_sim_clean.size(); ++i) {
        printf("%15.1f %15.1f %15.1f %15.1f\n",
               t_sim_clean[i].count() * 1000000, t_sim_A[i].count() * 1000000,
               t_sim_ct[i].count() * 1000000, t_sim_cp[i].count() * 1000000);
          }
    }
               
    
    std::string print() const {
      double s_cl{0.}, s_A{0.}, s_ct{0.}, s_cp{0.};
      for ( size_t i=1; i<t_sim_clean.size()-1; ++i) {
        s_cl += t_sim_clean[i].count();
        s_A  += t_sim_A[i].count();
        s_ct += t_sim_ct[i].count();
        s_cp += t_sim_cp[i].count();
      }
      
      double ss_cl{0.}, ss_A{0.}, ss_ct{0.}, ss_cp{0.};
      for ( size_t i=1; i<t_sim_clean.size()-1; ++i) {
        ss_cl += pow(t_sim_clean[i].count()-s_cl/(count-2),2);
        ss_A  += pow(t_sim_A[i].count()-s_A/(count-2),2);
        ss_ct += pow(t_sim_ct[i].count() - s_ct/(count-2),2);
        ss_cp += pow(t_sim_cp[i].count() - s_cp/(count-2),2);
      }

      ss_cl = 1000000 * sqrt(ss_cl / (count-2));
      ss_A  = 1000000 * sqrt(ss_A / (count-2));
      ss_ct = 1000000 * sqrt(ss_ct / (count-2));
      ss_cp = 1000000 * sqrt(ss_cp / (count-2));
    
      std::string out;
      char buf[100];
      sprintf(buf,"%12s %15s %15s %15s\n","kernel","total /s","avg launch /us", "std dev /us");
      out += buf;
      sprintf(buf,"%12s %15.8f %15.1f %15.1f\n","sim_clean", s_cl,
              s_cl * 1000000 / (count-2), ss_cl );
      out += buf;
      sprintf(buf,"%12s %15.8f %15.1f %15.1f\n","sim_A", s_A,
              s_A * 1000000 / (count-2), ss_A );
      out += buf;
      sprintf(buf,"%12s %15.8f %15.1f %15.1f\n","sim_ct", s_ct,
              s_ct * 1000000 / (count-2), ss_ct );
      out += buf;
      sprintf(buf,"%12s %15.8f %15.1f %15.1f\n","sim_cp", s_cp,
              s_cp * 1000000 / (count-2), ss_cp );
      out += buf;
      sprintf(buf,"%12s %15d +2\n","launch count",count-2);
      out += buf;

      return out;
    }
    
    friend std::ostream& operator<< (std::ostream& ost, const KernelTime& k) {
      return ost << k.print();
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

  CELL_CT_T*  ct{nullptr};
  int*        ct_h{nullptr};

  HitParams*    hitparams{nullptr};
  HitParams*    hitparams_h{nullptr};
  long*         simbins{nullptr};
  int           nbins{0};
  int           nsims{0};
  long          nhits{0};
  unsigned long ncells{0};
  unsigned int  maxhitct{0};
  float*        rand{nullptr};
  
} Sim_Args;

#endif
