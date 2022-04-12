/*
  Copyright (C) 2002-2021 CERN for the benefit of the ATLAS collaboration
*/

#include "CaloGpuGeneral_cu.h"
#include "GeoRegion.h"
#include "GeoGpu_structs.h"
#include "Hit.h"
#include "Rand4Hits.h"

//#include "gpuQ.h"
#include "Args.h"
#include <chrono>

#ifdef USE_KOKKOS
#include <Kokkos_Core.hpp>
#include <Kokkos_Random.hpp>
#endif

#define BLOCK_SIZE 256

#define M_PI 3.14159265358979323846
#define M_2PI 6.28318530717958647692

//using namespace CaloGpuGeneral_fnc;

namespace CaloGpuGeneral_cu {

//  __global__ void simulate_A( float E, int nhits, Chain0_Args args ) {
//
//    long t = threadIdx.x + blockIdx.x * blockDim.x;
//    if ( t < nhits ) {
//      Hit hit;
//      hit.E() = E;
//      CenterPositionCalculation_d( hit, args );
//      HistoLateralShapeParametrization_d( hit, t, args );
//      HitCellMappingWiggle_d( hit, args, t );
//    }
//  }
//
//  __global__ void simulate_ct( Chain0_Args args ) {
//
//    unsigned long tid = threadIdx.x + blockIdx.x * blockDim.x;
//    if ( tid < args.ncells ) {
//      if ( args.cells_energy[tid] > 0 ) {
//        unsigned int ct = atomicAdd( args.hitcells_ct, 1 );
//        Cell_E       ce;
//        ce.cellid           = tid;
//        ce.energy           = args.cells_energy[tid];
//        args.hitcells_E[ct] = ce;
//      }
//    }
//  }
//
//  __global__ void simulate_clean( Chain0_Args args ) {
//    unsigned long tid = threadIdx.x + blockIdx.x * blockDim.x;
//    if ( tid < args.ncells ) { args.cells_energy[tid] = 0.0; }
//    if ( tid == 0 ) args.hitcells_ct[0] = 0;
//  }
//
//  __host__ void simulate_A_cu( float E, int nhits, Chain0_Args& args ) {
//    int blocksize   = BLOCK_SIZE;
//    int threads_tot = nhits;
//    int nblocks     = ( threads_tot + blocksize - 1 ) / blocksize;
//    simulate_A<<<nblocks, blocksize>>>( E, nhits, args );
//  }

  void simulate_hits( float E, int nhits, Chain0_Args& args ) {

//    cudaError_t err = cudaGetLastError();
//
//    unsigned long ncells      = args.ncells;
//    int           blocksize   = BLOCK_SIZE;
//    int           threads_tot = args.ncells;
//    int           nblocks     = ( threads_tot + blocksize - 1 ) / blocksize;
//
//    simulate_clean<<<nblocks, blocksize>>>( args );
//    // 	cudaDeviceSynchronize() ;
//    // if (err != cudaSuccess) {
//    //       std::cout<< "simulate_clean "<<cudaGetErrorString(err)<< std::endl;
//    //}
//
//    blocksize   = BLOCK_SIZE;
//    threads_tot = nhits;
//    nblocks     = ( threads_tot + blocksize - 1 ) / blocksize;
//
//    //	 std::cout<<"Nblocks: "<< nblocks << ", blocksize: "<< blocksize
//    //               << ", total Threads: " << threads_tot << std::endl ;
//
//    //  int fh_size=args.fh2d_h.nbinsx+args.fh2d_h.nbinsy+2+(args.fh2d_h.nbinsx+1)*(args.fh2d_h.nbinsy+1) ;
//    // if(args.debug) std::cout<<"2DHisto_Func_size: " << args.fh2d_h.nbinsx << ", " << args.fh2d_h.nbinsy << "= " <<
//    // fh_size <<std::endl ;
//
//    simulate_A<<<nblocks, blocksize>>>( E, nhits, args );
//
//    //  cudaDeviceSynchronize() ;
//    //  err = cudaGetLastError();
//    // if (err != cudaSuccess) {
//    //        std::cout<< "simulate_A "<<cudaGetErrorString(err)<< std::endl;
//    //}
//
//    nblocks = ( ncells + blocksize - 1 ) / blocksize;
//    simulate_ct<<<nblocks, blocksize>>>( args );
//    //  cudaDeviceSynchronize() ;
//    // err = cudaGetLastError();
//    // if (err != cudaSuccess) {
//    //        std::cout<< "simulate_chain0_B1 "<<cudaGetErrorString(err)<< std::endl;
//    //}
//
//    int ct;
//    gpuQ( cudaMemcpy( &ct, args.hitcells_ct, sizeof( int ), cudaMemcpyDeviceToHost ) );
//    // std::cout<< "ct="<<ct<<std::endl;
//    gpuQ( cudaMemcpy( args.hitcells_E_h, args.hitcells_E, ct * sizeof( Cell_E ), cudaMemcpyDeviceToHost ) );
//
//    // pass result back
//    args.ct = ct;
//    //   args.hitcells_ct_h=hitcells_ct ;
  }

} // namespace CaloGpuGeneral_cu
