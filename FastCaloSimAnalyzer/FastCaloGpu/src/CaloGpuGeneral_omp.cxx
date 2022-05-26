/*
  Copyright (C) 2002-2021 CERN for the benefit of the ATLAS collaboration
*/

#include "CaloGpuGeneral_omp.h"
#include "GeoRegion.h"
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

  #pragma omp declare target
  inline void CenterPositionCalculation_dd( Hit& hit, const Chain0_Args args ) {

    //printf ( "Task being executed on host? %d!\n", omp_is_initial_device() );
    //printf ( "Num teams, threads: %d %d!\n", omp_get_num_teams(), omp_get_num_threads() ); //1467, 128
    hit.setCenter_r( ( 1. - args.extrapWeight ) * args.extrapol_r_ent + args.extrapWeight * args.extrapol_r_ext );
    hit.setCenter_z( ( 1. - args.extrapWeight ) * args.extrapol_z_ent + args.extrapWeight * args.extrapol_z_ext );
    hit.setCenter_eta( ( 1. - args.extrapWeight ) * args.extrapol_eta_ent + args.extrapWeight * args.extrapol_eta_ext );
    hit.setCenter_phi( ( 1. - args.extrapWeight ) * args.extrapol_phi_ent + args.extrapWeight * args.extrapol_phi_ext );
  }
  #pragma omp end declare target

  #pragma omp declare target
  void simulate_A( float E, int nhits, Chain0_Args args, Hit &hit ) {
    
    int m_default_device = omp_get_default_device();

    long t;
    const unsigned long ncells   = args.ncells;
    const unsigned long maxhitct = args.maxhitct;
    
    //declare mapper for members of struct
    {
      #pragma omp target //is_device_ptr( args.cells_energy ) device( m_default_device )
      #pragma omp teams distribute parallel for
      for ( t = 0; t < nhits; t++ ) {
  
//        Hit hit;
        hit.E() = E;
  
//        hit.setCenter_r( ( 1. - args.extrapWeight ) * args.extrapol_r_ent + args.extrapWeight * args.extrapol_r_ext );
//        hit.setCenter_z( ( 1. - args.extrapWeight ) * args.extrapol_z_ent + args.extrapWeight * args.extrapol_z_ext );
//        hit.setCenter_eta( ( 1. - args.extrapWeight ) * args.extrapol_eta_ent + args.extrapWeight * args.extrapol_eta_ext );
//        hit.setCenter_phi( ( 1. - args.extrapWeight ) * args.extrapol_phi_ent + args.extrapWeight * args.extrapol_phi_ext );
  	CenterPositionCalculation_dd( hit, args );

        HistoLateralShapeParametrization_d( hit, t, args );
	
        HitCellMappingWiggle_d( hit, args, t );
      }
    }

  }
  #pragma omp end declare target

  #pragma omp declare target
  void simulate_ct( Chain0_Args args ) {

    unsigned long tid;
    const unsigned long ncells   = args.ncells;
    const unsigned long maxhitct = args.maxhitct;
    
    #pragma omp target
    #pragma omp teams distribute parallel for
    for ( tid = 0; tid < ncells; tid++ ) {
      if ( args.cells_energy[tid] > 0 ) {
        //unsigned int ct = atomicAdd( args.hitcells_ct, 1 );
        unsigned int ct     = args.hitcells_ct[0];
        Cell_E                ce;
        ce.cellid           = tid;
        ce.energy           = args.cells_energy[tid];
        args.hitcells_E[ct] = ce;
        #pragma omp atomic update
          args.hitcells_ct[0]++;
      }
    }
  }
  #pragma omp end declare target

  #pragma omp declare target
  void simulate_clean( Chain0_Args args ) {
 
    int tid; 
    unsigned long ncells = args.ncells;
    
    #pragma omp target
    #pragma omp teams distribute parallel for
    for(tid = 0; tid < ncells; tid++) {
      args.cells_energy[tid] = 0.0;
      if ( tid == 0 ) args.hitcells_ct[tid] = 0;
    }
  }
  #pragma omp end declare target

  void simulate_hits( float E, int nhits, Chain0_Args& args ) {

    int m_default_device = omp_get_default_device();
    int m_initial_device = omp_get_initial_device();
    std::size_t m_offset = 0;

    const unsigned long ncells   = args.ncells;
    const unsigned long maxhitct = args.maxhitct;

    //TODO : args.hitcells_ct[0] = 0; //why does this give segfault
    //TODO : discuss memory allocation -- CPU or GPU? 18s vs 6s, correctness?

    #pragma omp target data map( alloc : args.hitcells_ct ) map( tofrom : args.hitcells_ct[:1] )\
                map( alloc : args.cells_energy ) map( tofrom : args.cells_energy[:ncells] ) 
                           //check if only 'from' performs better
    simulate_clean ( args );

    //TODO : discuss 'target data' faster than 'target'
    Hit hit;
    #pragma omp target data map(to : args.extrapol_eta_ent, args.extrapol_phi_ent, args.extrapol_r_ent,\
		    args.extrapol_z_ent, args.extrapol_eta_ext, args.extrapol_phi_ext, args.extrapol_r_ext,\
		    args.extrapol_z_ext, args.extrapWeight, args.charge, args.rand[:3*nhits],\
		    args.is_phi_symmetric, args.fh2d, args.fhs, args.geo, args.cs, args.nhits, hit,\
		    args.ncells ) map( alloc : args.cells_energy ) map( tofrom : args.cells_energy[:ncells] )
    simulate_A ( E, nhits, args, hit );

    //TODO : discuss 'target' faster than 'target data'
    #pragma omp target data map( alloc : args.hitcells_ct ) map( tofrom : args.hitcells_ct[:1] )\
                map( alloc : args.hitcells_E ) map( tofrom : args.hitcells_E[:maxhitct] )\
		map( alloc : args.cells_energy ) map( tofrom : args.cells_energy[:ncells] )
    simulate_ct ( args );

    int ct;// = args.hitcells_ct[0];
    if ( omp_target_memcpy( &ct, args.hitcells_ct, sizeof( int ),
                                    m_offset, m_offset, m_initial_device, m_default_device ) ) { 
      std::cout << "ERROR: copy hitcells_ct. " << std::endl;
    } 
    //gpuQ( cudaMemcpy( &ct, args.hitcells_ct, sizeof( int ), cudaMemcpyDeviceToHost ) );

    //if ( omp_target_memcpy( args.hitcells_E_h, args.hitcells_E, ct * sizeof( Cell_E ),
    //                                m_offset, m_offset, m_initial_device, m_default_device ) ) { 
    //  std::cout << "ERROR: copy hitcells_E_h. " << std::endl;
    //} 
    //gpuQ( cudaMemcpy( args.hitcells_E_h, args.hitcells_E, ct * sizeof( Cell_E ), cudaMemcpyDeviceToHost ) );


    // pass result back
    args.ct = ct;
    //   args.hitcells_ct_h=hitcells_ct ;
  }

} // namespace CaloGpuGeneral_omp
