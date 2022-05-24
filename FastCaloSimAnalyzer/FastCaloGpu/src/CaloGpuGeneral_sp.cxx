/*
  Copyright (C) 2002-2021 CERN for the benefit of the ATLAS collaboration
*/

#include <execution>
#include <algorithm>

#include "CaloGpuGeneral_sp.h"
#include "Rand4Hits.h"
#include "Hit.h"
#include "CountingIterator.h"
#include "nvToolsExt.h"

static CaloGpuGeneral::KernelTime timing;
static bool first{true};

using namespace CaloGpuGeneral_fnc;

namespace CaloGpuGeneral_stdpar {

  void simulate_clean(Chain0_Args& args) {

    std::for_each_n(std::execution::par_unseq, counting_iterator(0), args.ncells,
                    [=](unsigned int i) {
                      args.cells_energy[i] = 0;
                    }
                    );    
    
    args.hitcells_ct[0] = 0;
        
  }

  /* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

  void simulate_A( float E, int nhits, Chain0_Args args ) {

    // int* id = new int[10];
    // for (int i=0; i<10; ++i) {
    //   id[i] = i*10;
    // }

    // for (int i=0; i<10; ++i) {
    //   std::cout << i << "  " << id[i] << std::endl;
    // }

    // int* di{nullptr};
    // cudaMalloc( &di, 10*sizeof(int) );
    // cudaMemcpy( di, id, 10*sizeof(int), cudaMemcpyHostToDevice );
    // std::cout << "devptr: " << di << std::endl;
    
    // std::atomic<int> *ii = new std::atomic<int>{0};
    // std::cout << "sim_A: nhits: " << nhits << "  ii: " << *ii << std::endl;
    // float* ce = new float[200000];
    // ce[0] = 1.1;
    // ce[1] = 2.2;
    // std::for_each_n(std::execution::par_unseq, counting_iterator(0), 10,
    //                 [=](int i) {
    //                   int j = (*ii)++;
    //                   //                      int k = di[i];
    //                   //                      printf("%d %d %d %p\n",i,j,k, (void*)di);
    //                   printf("%d %d\n",i,j);
    //                   printf(" -> %p %f\n",(void*)ce,ce[i]);
    //                 } );
    // std::cout << "   after loop: " << *ii << std::endl;

    // float *fa = new float[10];
    // float *fb = new float[10];
    // float *fc = new float[10];
    // std::for_each_n(std::execution::par_unseq, counting_iterator(0), 10,
    //                 [=](int i) {
    //                   atomicAdd(&fa[i%2],float(i));
    //                   fb[i%2] += i;
    //                   atomicAdd(&fc[i],float(i));
    //                 } );
    // for (int i=0; i<10; ++i) {
    //   std::cout << i << " == " << fa[i] << "  " << fb[i] << "  " << fc[i] << std::endl;
    // }
    // return;

    std::for_each_n(std::execution::par_unseq, counting_iterator(0), nhits,
                  [=](unsigned int i) {

                    Hit hit;                    
                    hit.E() = E;
                    
                    CenterPositionCalculation_d( hit, args );

                    HistoLateralShapeParametrization_d( hit, i, args );
                    
                    HitCellMappingWiggle_d( hit, args, i );

                    
                  }
                    );
  }

  /* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

  void simulate_ct( Chain0_Args args ) {

    std::for_each_n(std::execution::par_unseq, counting_iterator(0), args.ncells,
                    [=](unsigned int i) {
                      // printf("ct: %p %p\n",(void*)args.hitcells_ct,(void*)args.hitcells_E);
                      if ( args.cells_energy[i] > 0 ) {
                        unsigned int ct = (*(args.hitcells_ct))++;
                        Cell_E              ce;
                        ce.cellid           = i;
                      #ifdef _NVHPC_STDPAR_NONE
                        ce.energy           = args.cells_energy[i];
                      #else
                        ce.energy           = args.cells_energy[i];
                        // ce.energy           = double(args.cells_energy[i])/CELL_ENE_FAC;
                      #endif
                        // ce.energy           = args.cells_energy[i];
                        args.hitcells_E[ct] = ce;
                        
                      }
                    } );
  }  
  
  /* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
  
  void simulate_hits( float E, int nhits, Chain0_Args& args ) {

    nvtxRangeId_t r1 = nvtxRangeStartA("sim_clean");

    auto t0 = std::chrono::system_clock::now();
    simulate_clean( args );

    nvtxRangeEnd(r1);
    nvtxRangeId_t r2 = nvtxRangeStartA("sim_A");

    auto t1 = std::chrono::system_clock::now();
    simulate_A( E, nhits, args );

    nvtxRangeEnd(r2);
    nvtxRangeId_t r3 = nvtxRangeStartA("sim_ct");

    auto t2 = std::chrono::system_clock::now();
    simulate_ct( args );

    nvtxRangeEnd(r3);
    auto t3 = std::chrono::system_clock::now();
    nvtxRangeId_t r4 = nvtxRangeStartA("sim_cp part 1");

    // pass result back
    args.ct = *args.hitcells_ct;

    nvtxRangeEnd(r4);
    nvtxRangeId_t r5 = nvtxRangeStartA("sim_cp part 2");

    std::memcpy( args.hitcells_E_h, args.hitcells_E, args.ct * sizeof( Cell_E ));

    nvtxRangeEnd(r5);

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
      std::cout << timing;
    } else {
      std::cout << "no kernel timing available" << std::endl;
    }

  }

} // namespace CaloGpuGeneral_stdpar
