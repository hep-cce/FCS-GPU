/*
  Copyright (C) 2002-2021 CERN for the benefit of the ATLAS collaboration
*/

#include <execution>
#include <algorithm>

#include "CaloGpuGeneral_sp.h"
#include "Rand4Hits.h"
#include "Hit.h"
#include "CountingIterator.h"

static CaloGpuGeneral::KernelTime timing;

using namespace CaloGpuGeneral_fnc;

namespace CaloGpuGeneral_stdpar {

  void simulate_clean(Chain0_Args& args) {

    if (m_cells_ene == 0) {
      m_cells_ene = new float[args.ncells];
    }
    
    args.cells_energy = m_cells_ene;
    std::for_each_n(std::execution::par_unseq, counting_iterator(0), args.ncells,
                    [=](unsigned int i) {
                      args.cells_energy[i] = 0.;
                    }
                    );    
    
    memset( m_cells_ene, 0, args.ncells*sizeof(float) );
    // for (unsigned int i=0; i<args.ncells; ++i) {
    //   args.cells_energy[i] = 0.;
    // }

    args.hitcells_ct[0] = 0;
        
  }

  /* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

  void simulate_A( float E, int nhits, Chain0_Args args ) {

    // std::cout << "sim_A: nhits: " << nhits << std::endl;

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
    
    
    std::for_each_n(std::execution::par_unseq, counting_iterator(0), nhits,
                  [=](unsigned int i) {
                    
                    Hit hit;                    
                    hit.E() = E;

                    // (*ii)++;
                    
                    CenterPositionCalculation_d( hit, args );
                    //                    printf("done CPC\n");
                    HistoLateralShapeParametrization_d( hit, i, args );
                    //                    printf("done HLSP %d\n",i);
                    HitCellMappingWiggle_d( hit, args, i );
                    // printf("done HCMW\n");
                  }
                  );
    //    std::cout << "===> done simulate_A " << *ii << "\n";
  }

  /* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

  void simulate_ct( Chain0_Args args ) {

    //    std::cout << "start sim_ct\n";
    std::for_each_n(std::execution::par_unseq, counting_iterator(0), args.ncells,
                    [=](unsigned int i) {
                      // printf("ct: %p %p\n",(void*)args.hitcells_ct,(void*)args.hitcells_E);
                      if ( args.cells_energy[i] > 0 ) {
                        unsigned int ct = (*(args.hitcells_ct))++;
                        // printf("ct: %p %p\n",(void*)args.hitcells_ct,(void*)args.hitcells_E);
                        //                        *(args.hitcells_ct) += 1;
                        Cell_E              ce;
                        ce.cellid           = i;
                        ce.energy           = args.cells_energy[i];
                        args.hitcells_E[ct] = ce;
                        
                        //                        printf("i: %u  id: %lu  ene: %f\n",ct, ce.cellid, ce.energy);
                        
                      }
                    } );
    std::cout << "sim_ct nhitcells: " << *(args.hitcells_ct) << std::endl;
    //    std::cout << "===> done simulate_ct\n";
  }  
  
  /* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
  
  void simulate_hits( float E, int nhits, Chain0_Args& args ) {

    auto t0 = std::chrono::system_clock::now();
    simulate_clean( args );

    auto t1 = std::chrono::system_clock::now();
    simulate_A( E, nhits, args );

    auto t2 = std::chrono::system_clock::now();
    simulate_ct( args );

    auto t3 = std::chrono::system_clock::now();

    // pass result back
    args.ct = *args.hitcells_ct;
    std::memcpy( args.hitcells_E_h, args.hitcells_E, args.ct * sizeof( Cell_E ));

    auto t4 = std::chrono::system_clock::now();

    // std::cout << "hits: " << args.ct << "\n";
    // for (int i=0; i<args.ct; ++i) {
    //   std::cout << "  " << args.hitcells_E[i].cellid << " "
    //             << args.hitcells_E_h[i].energy << "\n";
    // }
    
    CaloGpuGeneral::KernelTime kt( t1 - t0, t2 - t1, t3 - t2, t4 - t3 );
    timing += kt;
    
  }


  /* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
  
  void Rand4Hits_finish( void* rd4h ) {
    
    if ( (Rand4Hits*)rd4h ) delete (Rand4Hits*)rd4h;

    printf("time kernel sim_clean: %5.2f s / %4.0f us\n", timing.t_sim_clean.count(),
           timing.t_sim_clean.count() * 1000000 / timing.count);
    printf("time kernel sim_A:     %5.2f s / %4.0f us\n", timing.t_sim_A.count(),
           timing.t_sim_A.count() * 1000000 / timing.count);
    printf("time kernel sim_ct:    %5.2f s / %4.0f us\n", timing.t_sim_ct.count(),
           timing.t_sim_ct.count() * 1000000 / timing.count);
    printf("time kernel sim_cp:    %5.2f s / %4.0f us\n", timing.t_sim_cp.count(),
           timing.t_sim_cp.count() * 1000000 / timing.count);
    printf("time kernel count:     %5d\n",timing.count); 
  }

} // namespace CaloGpuGeneral_stdpar
