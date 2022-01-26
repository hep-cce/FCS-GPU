/*
  Copyright (C) 2002-2021 CERN for the benefit of the ATLAS collaboration
*/

#include <execution>
#include <algorithm>

#include "CaloGpuGeneral_sp.h"
#include "Rand4Hits.h"
#include "Hit.h"
#include "CountingIterator.h"


using namespace CaloGpuGeneral_fnc;

namespace CaloGpuGeneral_stdpar {

  void simulate_clean(Chain0_Args& args) {

    args.cells_energy = new float[args.ncells];
    for (unsigned int i=0; i<args.ncells; ++i) {
      args.cells_energy[i] = 0.;
    }
    args.hitcells_ct[0] = 0;
    // std::cout << "===> done simulate_clean\n";
        
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
                        // unsigned int ct = atomicAdd( args.hitcells_ct, 1 );
                        unsigned int ct = *(args.hitcells_ct);
                        // printf("ct: %p %p\n",(void*)args.hitcells_ct,(void*)args.hitcells_E);
                        *(args.hitcells_ct) += 1;
                        Cell_E              ce;
                        ce.cellid           = i;
                        ce.energy           = args.cells_energy[i];
                        args.hitcells_E[ct] = ce;
                      }
                    } );
    //    std::cout << "===> done simulate_ct\n";
  }  
  
  /* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
  
  void simulate_hits( float E, int nhits, Chain0_Args& args ) {

    simulate_clean( args );

    simulate_A( E, nhits, args );

    simulate_ct( args );


    int ct{0};
    // gpuQ( cudaMemcpy( &ct, args.hitcells_ct, sizeof( int ), cudaMemcpyDeviceToHost ) );
    // // std::cout<< "ct="<<ct<<std::endl;
    // gpuQ( cudaMemcpy( args.hitcells_E_h, args.hitcells_E, ct * sizeof( Cell_E ), cudaMemcpyDeviceToHost ) );

    // pass result back
    args.ct = ct;
    // //   args.hitcells_ct_h=hitcells_ct ;

  }

} // namespace CaloGpuGeneral_stdpar
