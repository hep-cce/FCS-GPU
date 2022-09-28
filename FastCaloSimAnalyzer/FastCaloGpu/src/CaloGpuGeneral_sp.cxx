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
#include "nvToolsExt.h"

static CaloGpuGeneral::KernelTime timing;
static bool first{true};

#define DO_ATOMIC_TESTS 0

using namespace CaloGpuGeneral_fnc;

namespace CaloGpuGeneral_stdpar {

  void test_atomicAdd_int() {
    std::cout << "---------- test_atomic<int>_add -------------\n";
    std::atomic<int> *ii = new std::atomic<int>{0};
    constexpr int N {10};
    std::for_each_n(std::execution::par_unseq, counting_iterator(0), N,
                    [=](int i) {
                      int j = (*ii)++;
                      printf("%d %d\n",i,j);
                    } );
    std::cout << "   after loop: " << *ii << " (should be " << N << ")" <<std::endl;
    std::cout << "---------- done test_atomic<int>_add -------------\n\n";
  }

  /* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

  void test_atomicAdd_float() {
    std::cout << "---------- test_atomicAdd_float -------------\n";
    constexpr int N {10};

    float ta[N]{0.}, tc[N]{0.};
    for (int i=0; i<N; ++i) {
      ta[i%2] += i;
      tc[i] += i;
    }

    
    float *fa = new float[N];
    float *fb = new float[N];
    float *fc = new float[N];
    std::for_each_n(std::execution::par_unseq, counting_iterator(0), N,
                    [=] (int i) {
                      fb[i%2] += i;
#if defined ( _NVHPC_STDPAR_NONE ) || defined ( _NVHPC_STDPAR_MULTICORE )
                      fa[i % 2] += i;
                      fc[i] += i;
#else
                      atomicAdd(&fa[i%2],float(i));
                      atomicAdd(&fc[i],float(i));
#endif
                    });
    for (int i=0; i<N; ++i) {
      printf("%d : %2g [%2g] %g  %g [%g]\n",i, fa[i], ta[i], fb[i], fc[i], tc[i]);
    }
    std::cout << "---------- done test_atomicAdd_float -------------\n\n";
  }

  /* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */  

  void simulate_clean(Sim_Args args) {
    nvtxRangeId_t r;
    if (!first) r = nvtxRangeStartA("sim_clean");

    // std::cout << "args.nsims: " << args.nsims << " args.ncells: " << args.ncells
    //           << "  MAX_SIM: "<< MAX_SIM
    //           << " ct: " << (void*)args.ct
    //           << " ene: " << (void*) args.cells_energy
    //           << std::endl;
    
    std::for_each_n(std::execution::par_unseq, counting_iterator(0), args.ncells*args.nsims,
                    [=](unsigned int tid) {
                      args.cells_energy[tid] = 0;
                      if ( tid < args.nsims ) args.ct[tid] = 0;  // faster than separate kernel
                    }
                    );
    // std::for_each_n(std::execution::par_unseq, counting_iterator(0), args.nsims,
    //                 [=](unsigned int tid) {
    //                   args.ct[tid] = 0;
    //                 }
    //                 );
    
    if (!first) nvtxRangeEnd(r);

  }

  /* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

  void simulate_hits_de( const Sim_Args args ) {

    nvtxRangeId_t r;
    if (!first) r = nvtxRangeStartA("sim_A");

    // std::atomic<int> *ii = new std::atomic<int>{0};
    std::for_each_n(std::execution::par_unseq, counting_iterator(0), args.nhits,
                  [=](unsigned int i) {

                    // int j = (*ii)++;
                    
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
    // int j = *ii;

    // if (j != args.nhits) {
    //   std::cout << "ERROR: loop not executed fully in simulate_hits_de. expected "
    //             << args.nhits << " got " << j << std::endl;
    // }

    if (!first) nvtxRangeEnd(r);
    
  }

  /* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

  void simulate_hits_ct( Sim_Args args ) {

    nvtxRangeId_t r;
    if (!first) r = nvtxRangeStartA("sim_ct");
    std::for_each_n(std::execution::par_unseq, counting_iterator(0), args.ncells*args.nsims,
                    [=](unsigned int tid) {

                      if ( args.cells_energy[tid] > 0 ) {

                        int           sim    = tid / args.ncells;
                        unsigned long cellid = tid % args.ncells;

#if defined(USE_ATOMICADD)
                        unsigned int ct = atomicAdd( &args.ct[sim], 1 );
#else
                        unsigned int ct = args.ct[sim]++;
#endif
                        Cell_E ce;
                        ce.cellid = cellid;
#if defined(_NVHPC_STDPAR_NONE) || defined (USE_ATOMICADD)
                        ce.energy = args.cells_energy[tid];
#else
                        ce.energy = double(args.cells_energy[tid])/CELL_ENE_FAC;
#endif
                        args.hitcells_E[ct + sim * MAXHITCT] = ce;
                        }

                    } );
    
    if (!first) nvtxRangeEnd(r);

  }  
  
  /* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
  
  void simulate_hits_gr( Sim_Args& args ) {

    if (DO_ATOMIC_TESTS) {
      test_atomicAdd_int();
      test_atomicAdd_float();
      return;
    }
    
    auto t0 = std::chrono::system_clock::now();
    simulate_clean( args );

    auto t1 = std::chrono::system_clock::now();
    simulate_hits_de( args );

    auto t2 = std::chrono::system_clock::now();
    simulate_hits_ct( args );

    auto t3 = std::chrono::system_clock::now();

    // pass result back
    nvtxRangeId_t r;
    if (!first) r = nvtxRangeStartA("sim_cp");
    std::memcpy( args.ct_h, args.ct, args.nsims * sizeof(int));
    // for (int i=0; i<args.nsims; ++i) {
    //   args.ct_h[i] = args.ct[i];
    // }
    std::memcpy( args.hitcells_E_h, args.hitcells_E, MAXHITCT * MAX_SIM * sizeof( Cell_E ));
    if (!first) nvtxRangeEnd(r);
    
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
    
    if (first) {
      first = false;
    } else {
      timing.add( t1 - t0, t2 - t1, t3 - t2, t4 - t3 );
    }
    
  }


  /* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
  
  void Rand4Hits_finish( void* rd4h ) {
    
    if ( (Rand4Hits*)rd4h ) delete (Rand4Hits*)rd4h;

    if (timing.count > 0) {
      std::cout << "kernel timing\n";
      std::cout << timing;
      // std::cout << "\n\n\n";
      // timing.printAll();
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
