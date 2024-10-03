/*
  Copyright (C) 2002-2021 CERN for the benefit of the ATLAS collaboration
*/

#include <execution>
#include <algorithm>

#include "CaloGpuGeneral_sp.h"
#include "Rand4Hits.h"
#include "Hit.h"
#include "CountingIterator.h"

// FIXME: Bug in nvhpc 24.X
#if defined ( _NVHPC_STDPAR_NONE )
  #include "nvToolsExt.h"
#endif

#define DO_ATOMIC_TESTS 0

static CaloGpuGeneral::KernelTime timing;
static bool first{true};

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

  void simulate_clean(Chain0_Args& args) {

    nvtxRangeId_t r;
    if (!first) r = nvtxRangeStartA("sim_clean");

    std::for_each_n(std::execution::par_unseq, counting_iterator(0), args.ncells,
                    [=](unsigned int i) {
                      args.cells_energy[i] = 0;
                    }
                    );    
    
    args.hitcells_ct[0] = 0;
    if (!first) nvtxRangeEnd(r);
        
  }

  /* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

  void simulate_A( float E, int nhits, Chain0_Args args ) {
    

    nvtxRangeId_t r;
    if (!first) r = nvtxRangeStartA("sim_A");

    // std::cout << "sim_A: nhits: " << nhits << std::endl;

    std::for_each_n(std::execution::par_unseq, counting_iterator(0), nhits,
                  [=](unsigned int i) {
                    
                    Hit hit;                    
                    hit.E() = E;
                    
                    CenterPositionCalculation_d( hit, args );

                    HistoLateralShapeParametrization_d( hit, i, args );
                    
                    HitCellMappingWiggle_d( hit, args, i );

                    
                  }
                    );
    
    if (!first) nvtxRangeEnd(r);

  }

  /* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

  void simulate_ct( Chain0_Args args ) {

    nvtxRangeId_t r;
    if (!first) r = nvtxRangeStartA("sim_ct");

    std::for_each_n(std::execution::par_unseq, counting_iterator(0), args.ncells,
                    [=](unsigned int i) {
                      // printf("ct: %p %p\n",(void*)args.hitcells_ct,(void*)args.hitcells_E);
                      if ( args.cells_energy[i] > 0 ) {
                      # if defined (USE_ATOMICADD)
                        unsigned int ct = atomicAdd( args.hitcells_ct, 1 );
                      # else
                        unsigned int ct = (*(args.hitcells_ct))++;
                      # endif
                        Cell_E              ce;
                        ce.cellid           = i;
                      #if defined (_NVHPC_STDPAR_NONE) || defined (USE_ATOMICADD)
                        ce.energy           = args.cells_energy[i];
                      #else
                        ce.energy           = double(args.cells_energy[i])/CELL_ENE_FAC;
                      #endif
                        // ce.energy           = args.cells_energy[i];
                        args.hitcells_E[ct] = ce;
                        
                      }
                    } );
    
    if (!first) nvtxRangeEnd(r);
  }  
  
  /* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
  
  void simulate_hits( float E, int nhits, Chain0_Args& args ) {

    if (DO_ATOMIC_TESTS) {
      test_atomicAdd_int();
      test_atomicAdd_float();
      return;
    }
    

    auto t0 = std::chrono::system_clock::now();
    simulate_clean( args );

    auto t1 = std::chrono::system_clock::now();
    simulate_A( E, nhits, args );

    auto t2 = std::chrono::system_clock::now();
    simulate_ct( args );

    auto t3 = std::chrono::system_clock::now();

    // pass result back
    nvtxRangeId_t r1,r2;
    if (!first) r1 = nvtxRangeStartA("sim_cp_1");
    args.ct = *args.hitcells_ct;
    if (!first) {
      nvtxRangeEnd(r1);
      r2 = nvtxRangeStartA("sim_cp_2");
    }
    std::memcpy( args.hitcells_E_h, args.hitcells_E, args.ct * sizeof( Cell_E ));
    if (!first) nvtxRangeEnd(r2);


    auto t4 = std::chrono::system_clock::now();

#ifdef DUMP_HITCELLS
    std::cout << "hitcells: " << args.ct << "  nhits: " << nhits << "  E: " << E << "\n";
    std::map<unsigned int,float> cm;
    for (int i=0; i<args.ct; ++i) {
      cm[args.hitcells_E_h[i].cellid] = args.hitcells_E_h[i].energy;
    }
    for (auto &em: cm) {
      std::cout << "  cell: " << em.first << "  " << em.second << std::endl;
    }
#endif
    
    timing.add( t1 - t0, t2 - t1, t3 - t2, t4 - t3 );

    if (first) {
      first = false;
    }
    
  }


  /* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
  
  void Rand4Hits_finish( void* rd4h ) {
    
    if ( (Rand4Hits*)rd4h ) delete (Rand4Hits*)rd4h;

    std::cout << timing;

  }

} // namespace CaloGpuGeneral_stdpar
