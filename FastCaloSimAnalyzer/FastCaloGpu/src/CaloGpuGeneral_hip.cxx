#include "hip/hip_runtime.h"
/*
  Copyright (C) 2002-2021 CERN for the benefit of the ATLAS collaboration
*/

#include "CaloGpuGeneral_cu.h"
#include "CaloGpuGeneral_al.h"
#include "GeoRegion.h"
#include "GeoGpu_structs.h"
#include "Hit.h"
#include "Rand4Hits.h"
//#include "helloWorld.h"

#include "gpuQ.h"
#include "Args.h"
#include <chrono>
#include <climits>

#ifdef USE_KOKKOS
#include <Kokkos_Core.hpp>
#include <Kokkos_Random.hpp>
#endif

#define BLOCK_SIZE 256

#define M_PI 3.14159265358979323846
#define M_2PI 6.28318530717958647692

using namespace CaloGpuGeneral_fnc;

static CaloGpuGeneral::KernelTime timing;

namespace CaloGpuGeneral_cu {

  __host__ void Rand4Hits_finish( void* rd4h ) {

    size_t free, total;
    gpuQ( hipMemGetInfo( &free, &total ) );
    std::cout << "GPU memory used(MB): " << ( total - free ) / 1000000
              << std::endl;
    //    if ( (Rand4Hits*)rd4h ) delete (Rand4Hits*)rd4h;

    std::cout << timing;
    
  }

  /* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

  __global__ void simulate_A( float E, int nhits, Chain0_Args args ) {

    long t = threadIdx.x + blockIdx.x * blockDim.x;
    if ( t < nhits ) {
      Hit hit;
      hit.E() = E;

      CenterPositionCalculation_d( hit, args );
      HistoLateralShapeParametrization_d( hit, t, args );
      HitCellMappingWiggle_d( hit, args, t );

    }
  }

  /* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

  __global__ void simulate_ct( Chain0_Args args ) {

    unsigned long tid = threadIdx.x + blockIdx.x * blockDim.x;
    if ( tid < args.ncells ) {
      if ( args.cells_energy[tid] > 0 ) {
        unsigned int ct = atomicAdd( args.hitcells_ct, 1 );
        Cell_E       ce;
        ce.cellid           = tid;
        ce.energy           = args.cells_energy[tid];
        args.hitcells_E[ct] = ce;
        // printf("i: %u  id: %lu  ene: %f\n", ct, ce.cellid, ce.energy);
      }
    }
  }

  /* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

  __global__ void simulate_clean( Chain0_Args args ) {
    unsigned long tid = threadIdx.x + blockIdx.x * blockDim.x;
    if ( tid < args.ncells ) { args.cells_energy[tid] = 0.0; }
    if ( tid == 0 ) args.hitcells_ct[0] = 0;
  }

  /* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

  __host__ void simulate_A_cu( float E, int nhits, Chain0_Args& args ) {
    int blocksize   = BLOCK_SIZE;
    int threads_tot = nhits;
    int nblocks     = ( threads_tot + blocksize - 1 ) / blocksize;
    hipLaunchKernelGGL(simulate_A, nblocks, blocksize, 0, 0,  E, nhits, args );
  }

  /* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

  __host__ void simulate_hits( float E, int nhits, Chain0_Args& args ) {

    //    printf("nhits: %d   ene: %f\n",nhits,E);
    
    hipError_t err = hipGetLastError();

    unsigned long ncells      = args.ncells;
    int           blocksize   = BLOCK_SIZE;
    int           threads_tot = args.ncells;
    int           nblocks     = ( threads_tot + blocksize - 1 ) / blocksize;

    
    auto              t0   = std::chrono::system_clock::now();
    hipLaunchKernelGGL(simulate_clean, nblocks, blocksize, 0, 0,  args );
    gpuQ( hipGetLastError() );
    gpuQ( hipDeviceSynchronize() );

    blocksize   = BLOCK_SIZE;
    threads_tot = nhits;
    nblocks     = ( threads_tot + blocksize - 1 ) / blocksize;
    auto t1     = std::chrono::system_clock::now();
    hipLaunchKernelGGL(simulate_A, nblocks, blocksize, 0, 0,  E, nhits, args );
    gpuQ( hipGetLastError() );
    gpuQ( hipDeviceSynchronize() );


    nblocks = ( ncells + blocksize - 1 ) / blocksize;
    auto t2 = std::chrono::system_clock::now();
    hipLaunchKernelGGL(simulate_ct, nblocks, blocksize, 0, 0,  args );
    gpuQ( hipGetLastError() );
    gpuQ( hipDeviceSynchronize() );

    int ct;
    auto t3 = std::chrono::system_clock::now();
    gpuQ( hipMemcpy( &ct, args.hitcells_ct, sizeof( int ), hipMemcpyDeviceToHost ) );
    gpuQ( hipMemcpy( args.hitcells_E_h, args.hitcells_E, ct * sizeof( Cell_E ), hipMemcpyDeviceToHost ) );

    auto t4 = std::chrono::system_clock::now();
    // pass result back
    args.ct = ct;
    //   args.hitcells_ct_h=hitcells_ct ;

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
    
  }

} // namespace CaloGpuGeneral_cu
