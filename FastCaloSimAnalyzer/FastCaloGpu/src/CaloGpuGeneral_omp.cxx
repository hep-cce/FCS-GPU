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

  void simulate_hits_master( float E, int nhits, Chain0_Args args ) {
 
    const unsigned long ncells = args.ncells;
    const unsigned long maxhitct = args.maxhitct;
    auto cells_energy = args.cells_energy;
    auto hitcells_ct  = args.hitcells_ct;
    int m_default_device = omp_get_default_device();

    /************* clean **********/	  

    int tid; 
    #pragma omp target is_device_ptr ( cells_energy, hitcells_ct )            
    #pragma omp teams distribute parallel for
    for(tid = 0; tid < ncells; tid++) {
      cells_energy[tid] = 0.0;
      if ( tid == 0 ) hitcells_ct[tid] = 0;
    }
   
    /************* A **********/

    long t;
    Hit hit;
    #pragma omp target data map(to : args.extrapol_eta_ent, args.extrapol_phi_ent, args.extrapol_r_ent,\
                  args.extrapol_z_ent, args.extrapol_eta_ext, args.extrapol_phi_ext, args.extrapol_r_ext,\
                  args.extrapol_z_ext, args.extrapWeight, args.charge, args.rand[:3*nhits],\
                  args.is_phi_symmetric, args.fh2d, args.fhs, args.geo, args.cs, args.nhits, hit,\
                  args.ncells )
    //declare mapper for members of struct
    {
      #pragma omp target is_device_ptr( cells_energy )
      #pragma omp teams distribute parallel for
      for ( t = 0; t < nhits; t++ ) {
        //Hit hit;
        hit.E() = E;
	//printf("num teams = %d", omp_get_num_teams() );
  
      //CenterPositionCalculation_d( hit, args );
        hit.setCenter_r( ( 1. - args.extrapWeight ) * args.extrapol_r_ent + args.extrapWeight * args.extrapol_r_ext );
        hit.setCenter_z( ( 1. - args.extrapWeight ) * args.extrapol_z_ent + args.extrapWeight * args.extrapol_z_ext );
        hit.setCenter_eta( ( 1. - args.extrapWeight ) * args.extrapol_eta_ent + args.extrapWeight * args.extrapol_eta_ext );
        hit.setCenter_phi( ( 1. - args.extrapWeight ) * args.extrapol_phi_ent + args.extrapWeight * args.extrapol_phi_ext );
 
      //HistoLateralShapeParametrization_d( hit, t, args );
        // int     pdgId    = args.pdgId;
        float charge = args.charge;
    
        // int cs=args.charge;
        float center_eta = hit.center_eta();
        float center_phi = hit.center_phi();
        float center_r   = hit.center_r();
        float center_z   = hit.center_z();
    
        float alpha, r, rnd1, rnd2;
        rnd1 = args.rand[t];
        rnd2 = args.rand[t + args.nhits];
    
        if ( args.is_phi_symmetric ) {
          if ( rnd2 >= 0.5 ) { // Fill negative phi half of shape
            rnd2 -= 0.5;
            rnd2 *= 2;
            rnd_to_fct2d( alpha, r, rnd1, rnd2, args.fh2d );
            alpha = -alpha;
          } else { // Fill positive phi half of shape
            rnd2 *= 2;
            rnd_to_fct2d( alpha, r, rnd1, rnd2, args.fh2d );
          }
        } else {
          rnd_to_fct2d( alpha, r, rnd1, rnd2, args.fh2d );
        }
    
        float delta_eta_mm = r * cos( alpha );
        float delta_phi_mm = r * sin( alpha );
    
        // Particles with negative eta are expected to have the same shape as those with positive eta after transformation:
        // delta_eta --> -delta_eta
        if ( center_eta < 0. ) delta_eta_mm = -delta_eta_mm;
        // Particle with negative charge are expected to have the same shape as positively charged particles after
        // transformation: delta_phi --> -delta_phi
        if ( charge < 0. ) delta_phi_mm = -delta_phi_mm;
       
        //TODO : save exp and divisions
        float dist000    = sqrt( center_r * center_r + center_z * center_z );
        float eta_jakobi = abs( 2.0 * exp( -center_eta ) / ( 1.0 + exp( -2 * center_eta ) ) );
    
        float delta_eta = delta_eta_mm / eta_jakobi / dist000;
        float delta_phi = delta_phi_mm / center_r;
    
        hit.setEtaPhiZE( center_eta + delta_eta, center_phi + delta_phi, center_z, hit.E() );
 



      //HitCellMappingWiggle_d( hit, args, t, cells_energy );
        int    nhist        = ( *( args.fhs ) ).nhist;
        float* bin_low_edge = ( *( args.fhs ) ).low_edge;
    
        float eta = fabs( hit.eta() );
        if ( eta < bin_low_edge[0] || eta > bin_low_edge[nhist] ) { HitCellMapping_d( hit, t, args, cells_energy ); }
    
        int bin = nhist;
        for ( int i = 0; i < nhist + 1; ++i ) {
          if ( bin_low_edge[i] > eta ) {
            bin = i;
            break;
          }
        }
    
        //  bin=find_index_f(bin_low_edge, nhist+1, eta ) ;
    
        bin -= 1;
    
        unsigned int mxsz       = args.fhs->mxsz;
        uint32_t*    contents   = &( args.fhs->d_contents1D[bin * mxsz] );
        float*       borders    = &( args.fhs->d_borders1D[bin * mxsz] );
        int          h_size     = ( *( args.fhs ) ).h_szs[bin];
        uint32_t     s_MaxValue = ( *( args.fhs ) ).s_MaxValue;
    
        float rnd = args.rand[t + 2 * args.nhits];
    
        float wiggle = rnd_to_fct1d( rnd, contents, borders, h_size, s_MaxValue );
    
        float hit_phi_shifted = hit.phi() + wiggle;
        hit.phi()             = Phi_mpi_pi( hit_phi_shifted );
   
      //HitCellMapping	
        long long cellele = getDDE( args.geo, args.cs, hit.eta(), hit.phi() );

        #pragma omp atomic update
        //*( cells_energy + cellele ) += E;
        cells_energy[cellele] += (float)(E); //typecast is necessary

      }
    }

    /************* ct ***********/

    #pragma omp target is_device_ptr ( cells_energy, hitcells_ct )            
    #pragma omp teams distribute parallel for
    for(tid = 0; tid < ncells; tid++) {
      cells_energy[tid] = 0.0;
      if ( tid == 0 ) hitcells_ct[tid] = 0;
    }


  }


  void simulate_A( float E, int nhits, Chain0_Args args ) {
    
    int m_default_device = omp_get_default_device();

    long t;
    const unsigned long ncells   = args.ncells;
    const unsigned long maxhitct = args.maxhitct;
    
    auto cells_energy = args.cells_energy;
    Hit hit;
    //declare mapper for members of struct
    #pragma omp target data map(to : args.extrapol_eta_ent, args.extrapol_phi_ent, args.extrapol_r_ent,\
                  args.extrapol_z_ent, args.extrapol_eta_ext, args.extrapol_phi_ext, args.extrapol_r_ext,\
                  args.extrapol_z_ext, args.extrapWeight, args.charge, args.rand[:3*nhits],\
                  args.is_phi_symmetric, args.fh2d, args.fhs, args.geo, args.cs, args.nhits, hit,\
                  args.ncells, cells_energy[:ncells] )
    {
      #pragma omp target //is_device_ptr( cells_energy )
      #pragma omp teams distribute parallel for
      for ( t = 0; t < nhits; t++ ) {
//        Hit hit;
        hit.E() = E;
  
	CenterPositionCalculation_d( hit, args );

        HistoLateralShapeParametrization_d( hit, t, args );
	
        HitCellMappingWiggle_d( hit, args, t, cells_energy );

      }
    }

  }

  void simulate_ct( Chain0_Args args ) {

    unsigned long tid;
    const unsigned long ncells   = args.ncells;
    const unsigned long maxhitct = args.maxhitct;
    
    auto cells_energy = args.cells_energy;
    auto hitcells_ct  = args.hitcells_ct;
    auto hitcells_E   = args.hitcells_E;

    #pragma omp target is_device_ptr ( cells_energy, hitcells_ct, hitcells_E ) 
    #pragma omp teams distribute parallel for
    for ( tid = 0; tid < ncells; tid++ ) {
      if ( args.cells_energy[tid] > 0 ) {
        //unsigned int ct = atomicAdd( args.hitcells_ct, 1 );
        unsigned int ct     = hitcells_ct[0];
        Cell_E                ce;
        ce.cellid           = tid;
        ce.energy           = cells_energy[tid];
        hitcells_E[ct]      = ce;
        #pragma omp atomic update
          hitcells_ct[0]++;
      }
    }
  }

  void simulate_clean( Chain0_Args args ) {
 
    int tid; 
    const unsigned long ncells = args.ncells;

    auto cells_energy = args.cells_energy;
    auto hitcells_ct  = args.hitcells_ct;

    #pragma omp target is_device_ptr ( cells_energy, hitcells_ct )            
    #pragma omp teams distribute parallel for
    for(tid = 0; tid < ncells; tid++) {
      cells_energy[tid] = 0.0;
      if ( tid == 0 ) hitcells_ct[tid] = 0;
    }
  }

  void simulate_hits( float E, int nhits, Chain0_Args& args ) {

    int m_default_device = omp_get_default_device();
    int m_initial_device = omp_get_initial_device();
    std::size_t m_offset = 0;

    const unsigned long ncells   = args.ncells;
    const unsigned long maxhitct = args.maxhitct;
  
    //TODO : args.hitcells_ct[0] = 0; //why does this give segfault
    //TODO : discuss memory allocation -- CPU or GPU? 18s vs 6s, correctness?

    simulate_hits_master ( E, nhits, args );
    
    //TODO : discuss 'target data' faster than 'target'
    //simulate_clean ( args );
    //simulate_A ( E, nhits, args );
    //simulate_ct ( args );

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
