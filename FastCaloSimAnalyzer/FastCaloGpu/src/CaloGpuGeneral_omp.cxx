/*
  Copyright (C) 2002-2021 CERN for the benefit of the ATLAS collaboration
*/

#include "CaloGpuGeneral_omp.h"
//#include "GeoRegion.h"
#include "GeoGpu_structs.h"
#include "Hit.h"
#include "Rand4Hits.h"

//#include "gpuQ.h"
#include "Args.h"
#include <chrono>
#include <iostream>
#include <omp.h>

#ifdef USE_KOKKOS
#include <Kokkos_Core.hpp>
#include <Kokkos_Random.hpp>
#endif

#define BLOCK_SIZE 256

#define M_PI 3.14159265358979323846
#define M_2PI 6.28318530717958647692

using namespace CaloGpuGeneral_fnc;

namespace CaloGpuGeneral_omp {

//  inline void CenterPositionCalculation_d( Hit& hit, const Chain0_Args args ) {
//
//    printf ( "Task being executed on host? %d %d %d!\n", omp_is_initial_device(), omp_get_num_teams(), omp_get_num_threads() ); //1467, 128
//    hit.setCenter_r( ( 1. - args.extrapWeight ) * args.extrapol_r_ent + args.extrapWeight * args.extrapol_r_ext );
//    hit.setCenter_z( ( 1. - args.extrapWeight ) * args.extrapol_z_ent + args.extrapWeight * args.extrapol_z_ext );
//    hit.setCenter_eta( ( 1. - args.extrapWeight ) * args.extrapol_eta_ent + args.extrapWeight * args.extrapol_eta_ext );
//    hit.setCenter_phi( ( 1. - args.extrapWeight ) * args.extrapol_phi_ent + args.extrapWeight * args.extrapol_phi_ext );
//  }


  
  void simulate_A( float E, int nhits, Chain0_Args args ) {

    int t;
   
    #pragma omp target map(args.hitcells_ct[:1]) 
    {
      for(t = 0; t < nhits; t++) {
        Hit hit;
        hit.E() = E;
        //CenterPositionCalculation_d( hit, args );

      }
    }

    //long t = threadIdx.x + blockIdx.x * blockDim.x;
    //if ( t < nhits ) {
    //  Hit hit;
    //  hit.E() = E;
    //  CenterPositionCalculation_d( hit, args );
    //  HistoLateralShapeParametrization_d( hit, t, args );
    //  HitCellMappingWiggle_d( hit, args, t );
    //}
  }

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
  void simulate_clean( Chain0_Args args ) {
 
    int tid; 
    unsigned long ncells = args.ncells;
    //std::cout << "ncells = " << args.ncells << std::endl;  
    /*TODO: Where is args.cells_energy allocated? */
    
    #pragma omp target map(args.hitcells_ct[:1]) map(args.cells_energy[:ncells]) 
    {
      #pragma omp teams distribute parallel for
      for(tid = 0; tid < ncells; tid++) {
        args.cells_energy[tid] = 0.0;
        if ( tid == 0 ) args.hitcells_ct[0] = 0;
        //printf ( "Task being executed on host? %d %d %d!\n", omp_is_initial_device(), omp_get_num_teams(), omp_get_num_threads() ); //1467, 128
      }
    }

  }

  void simulate_hits( float E, int nhits, Chain0_Args& args ) {

    simulate_clean ( args );

    //simulate_A<<<nblocks, blocksize>>>( E, nhits, args );
    simulate_A ( E, nhits, args );

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

} // namespace CaloGpuGeneral_omp
