/*
  Copyright (C) 2002-2021 CERN for the benefit of the ATLAS collaboration
*/

#include <execution>
#include <algorithm>

#include "CaloGpuGeneral_sp.h"
#include "Rand4Hits.h"
#include "Hit.h"
#include "CountingIterator.h"
#include "GpuParams.h"

static CaloGpuGeneral::KernelTime timing;
static bool first{true};

using namespace CaloGpuGeneral_fnc;

namespace CaloGpuGeneral_stdpar {

  void simulate_clean(Sim_Args args) {

    std::for_each_n(std::execution::par_unseq, counting_iterator(0), args.ncells*args.nsims,
                    [=](unsigned int tid) {
                      args.cells_energy[tid] = 0.0;
                      if ( tid < args.nsims ) args.ct[tid] = 0;
                    }
                    );     
  }

  /* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

  void simulate_hits_de( const Sim_Args args ) {

    std::atomic<int> *ii = new std::atomic<int>{0};
    std::for_each_n(std::execution::par_unseq, counting_iterator(0), args.nhits,
                  [=](unsigned int i) {

                    int j = (*ii)++;
                    
                    Hit hit;
                    
                    int bin = find_index_long( args.simbins, args.nbins, i );
                    HitParams hp = args.hitparams[bin];
                    hit.E() = hp.E;
                    
                    CenterPositionCalculation_g_d( hp, hit, i, args );
                    HistoLateralShapeParametrization_g_d( hp, hit, i, args );
                    if ( hp.cmw ) HitCellMappingWiggle_g_d( hp, hit, i, args );
                    HitCellMapping_g_d( hp, hit, i, args );
                    
                  }
                    );
    int j = *ii;

    if (j != args.nhits) {
      std::cout << "ERROR: loop not executed fully in simulate_hits_de. expected "
                << args.nhits << " got " << j << std::endl;
    }
    
  }

  /* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

  void simulate_hits_ct( Sim_Args args ) {

    std::for_each_n(std::execution::par_unseq, counting_iterator(0), args.ncells*args.nsims,
                    [=](unsigned int tid) {

                      if ( args.cells_energy[tid] > 0 ) {

                        int           sim    = tid / args.ncells;
                        unsigned long cellid = tid % args.ncells;
                        
                        unsigned int ct = args.ct[sim]++;
                        Cell_E ce;
                        ce.cellid = cellid;
                      #ifdef _NVHPC_STDPAR_NONE
                        ce.energy = args.cells_energy[tid];
                      #else
                        ce.energy = double(args.cells_energy[tid])/CELL_ENE_FAC;
                      #endif
                        ce.energy                            = args.cells_energy[tid];
                        args.hitcells_E[ct + sim * MAXHITCT] = ce;
                        }

                    } );
  }  
  
  /* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
  
  void simulate_hits_gr( Sim_Args& args ) {

    auto t0 = std::chrono::system_clock::now();
    simulate_clean( args );

    auto t1 = std::chrono::system_clock::now();
    simulate_hits_de( args );

    auto t2 = std::chrono::system_clock::now();
    simulate_hits_ct( args );

    auto t3 = std::chrono::system_clock::now();

    // pass result back
    //    std::memcpy( args.ct_h, args.ct, args.nsims * sizeof(int));
    for (int i=0; i<args.nsims; ++i) {
      args.ct_h[i] = args.ct[i];
    }
    std::memcpy( args.hitcells_E_h, args.hitcells_E, MAXHITCT * MAX_SIM * sizeof( Cell_E ));
    auto t4 = std::chrono::system_clock::now();

#ifdef DUMP_HITCELLS
    std::cout << "nsim: " << args.nsims << "\n";
    for (int isim=0; isim<args.nsims; ++isim) {
      std::cout << "  nhit: " << args.ct_h[isim] << "\n";
      std::map<unsigned int,float> cm;
      for (int ihit=0; ihit<args.ct_h[isim]; ++ihit) {
        cm[args.hitcells_E_h[ihit+isim*MAXHITCT].cellid] = args.hitcells_E_h[ihit+isim*MAXHITCT].energy;
      }

      int i=0;
      for (auto &em: cm) {
        std::cout << "   " << isim << " " << i++ << "  cell: " << em.first << "  " << em.second << std::endl;
      }
    }
#endif
    
    CaloGpuGeneral::KernelTime kt( t1 - t0, t2 - t1, t3 - t2, t4 - t3 );
    if (first) {
      first = false;
    } else {
      timing += kt;
    }
    
  }


  /* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
  
  void Rand4Hits_finish( void* rd4h ) {
    
    if ( (Rand4Hits*)rd4h ) delete (Rand4Hits*)rd4h;

    if (timing.count > 0) {
      std::cout << "kernel timing\n";
      printf("%12s %15s %15s\n","kernel","total /s","avg launch /us");
      printf("%12s %15.8f %15.1f\n","sim_clean",timing.t_sim_clean.count(),
             timing.t_sim_clean.count() * 1000000 /timing.count);
      printf("%12s %15.8f %15.1f\n","sim_A",timing.t_sim_A.count(),
             timing.t_sim_A.count() * 1000000 /timing.count);
      printf("%12s %15.8f %15.1f\n","sim_ct",timing.t_sim_ct.count(),
             timing.t_sim_ct.count() * 1000000 /timing.count);
      printf("%12s %15.8f %15.1f\n","sim_cp",timing.t_sim_cp.count(),
             timing.t_sim_cp.count() * 1000000 /timing.count);
      printf("%12s %15d\n","launch count",timing.count);
    } else {
      std::cout << "no kernel timing available" << std::endl;
    }

  }

  /* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

  void load_hitsim_params( void* rd4h, HitParams* hp, long* simbins, int bins ) {

  if ( !(Rand4Hits*)rd4h ) {
    std::cout << "Error load hit simulation params ! ";
    exit( 2 );
  }

  HitParams* hp_g      = ( (Rand4Hits*)rd4h )->get_hitparams();
  long*      simbins_g = ( (Rand4Hits*)rd4h )->get_simbins();

  std::memcpy( hp_g, hp, bins * sizeof( HitParams ) );
  std::memcpy( simbins_g, simbins, bins * sizeof( long ) );
}


} // namespace CaloGpuGeneral_stdpar
