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
#define GRID_SIZE 734
#define GRID_SIZE1 5

#define M_PI 3.14159265358979323846
#define M_2PI 6.28318530717958647692

using namespace CaloGpuGeneral_fnc;

namespace CaloGpuGeneral_omp {

  inline void simulate_A( float E, int nhits, Chain0_Args args ) {

    const unsigned long ncells   = args.ncells;
    const unsigned long maxhitct = args.maxhitct;
    
    auto cells_energy = args.cells_energy;
    auto hitcells_ct  = args.hitcells_ct;
    auto rand         = args.rand;
    auto geo          = args.geo;

    int m_initial_device = omp_get_initial_device();

    /************* A **********/

    long t;
    //Hit hit;
    #pragma omp target is_device_ptr( cells_energy, rand, geo ) //map( to : args )
    //#pragma omp target is_device_ptr( cells_energy, rand, geo ) nowait //depend( in : args )
    {
      #pragma omp teams distribute parallel for num_teams(60) //num_threads(64) //num_teams default 33
      for ( t = 0; t < nhits; t++ ) {
        Hit hit;
        hit.E() = E;
  
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
        rnd1 = rand[t];
        rnd2 = rand[t + args.nhits];
        // printf ( " rands are %f %f ----> \n ", rnd1, rnd2);  
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
        if ( eta < bin_low_edge[0] || eta > bin_low_edge[nhist] ) { 
           //HitCellMapping_d( hit, t, args, cells_energy ); 
	     long long cellele = getDDE( args.geo, args.cs, hit.eta(), hit.phi() );
	     #pragma omp atomic update
             cells_energy[cellele] += (float)(E);
	}
    
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
    
        float rnd = rand[t + 2 * args.nhits];
    
        float wiggle = rnd_to_fct1d( rnd, contents, borders, h_size, s_MaxValue );
    
        float hit_phi_shifted = hit.phi() + wiggle;
        hit.phi()             = Phi_mpi_pi( hit_phi_shifted );
   
      //HitCellMapping	
        long long cellele = getDDE( args.geo, args.cs, hit.eta(), hit.phi() );
        // printf("t = %ld cellee %lld hit.eta %f hit.phi %f \n", t, cellele, hit.eta(), hit.phi());

        #pragma omp atomic update
        //*( cells_energy + cellele ) += E;
        cells_energy[cellele] += (float)(E); //typecast is necessary

      }
    }
    //#pragma omp taskwait

  }

  inline void simulate_ct( Chain0_Args args ) {

    const unsigned long ncells   = args.ncells;
    
    auto cells_energy = args.cells_energy;
    auto hitcells_ct  = args.hitcells_ct;
    auto hitcells_E   = args.hitcells_E;
    
    #pragma omp target is_device_ptr ( cells_energy, hitcells_ct, hitcells_E ) //nowait
    #pragma omp teams distribute parallel for num_teams(GRID_SIZE) num_threads(BLOCK_SIZE) //thread_limit(128) //num_teams default 1467, threads default 128
    for ( int tid = 0; tid < ncells; tid++ ) {
      if ( cells_energy[tid] > 0. ) {
	 unsigned int ct;
         #pragma omp atomic capture
	 ct = hitcells_ct[0]++; 
	      
	 Cell_E                ce;
         ce.cellid           = tid;
         ce.energy           = cells_energy[tid];
         hitcells_E[ct]      = ce;
         //printf ( "ct %d %d energy %f cellid %d \n", ct, hitcells_ct[0], hitcells_E[ct].energy, hitcells_E[ct].cellid);
      }
    }
    //#pragma omp taskwait

  }

  inline void simulate_clean( Chain0_Args& args ) {
 
    auto cells_energy = args.cells_energy;
    auto hitcells_ct  = args.hitcells_ct;

    const unsigned long ncells   = args.ncells;

    int tid; 
    #pragma omp target is_device_ptr ( cells_energy, hitcells_ct ) //nowait
    #pragma omp teams distribute parallel for num_teams(GRID_SIZE) num_threads(BLOCK_SIZE) // num_teams default 1467, threads default 128
    for(tid = 0; tid < ncells; tid++) {
      //printf(" num teams = %d, num threads = %d", omp_get_num_teams(), omp_get_num_threads() );
      cells_energy[tid] = 0.;
      //hitcells_ct[0] = 0;
      if ( tid == 0 ) hitcells_ct[tid] = 0;
    }

  }

  void simulate_hits( float E, int nhits, Chain0_Args& args, int select_device ) {

    int m_initial_device = omp_get_initial_device();
    std::size_t m_offset = 0;

    simulate_clean ( args );
    simulate_A ( E, nhits, args );
    simulate_ct ( args );
    
    int *ct = (int *) malloc( sizeof( int ) );
    if ( omp_target_memcpy( ct, args.hitcells_ct, sizeof( int ),
                                    m_offset, m_offset, m_initial_device, select_device ) ) { 
      std::cout << "ERROR: copy hitcells_ct. " << std::endl;
    } 
    //gpuQ( cudaMemcpy( &ct, args.hitcells_ct, sizeof( int ), cudaMemcpyDeviceToHost ) );

    if ( omp_target_memcpy( args.hitcells_E_h, args.hitcells_E, ct[0] * sizeof( Cell_E ),
                                    m_offset, m_offset, m_initial_device, select_device ) ) { 
      std::cout << "ERROR: copy hitcells_E_h. " << std::endl;
    } 
    //gpuQ( cudaMemcpy( args.hitcells_E_h, args.hitcells_E, ct * sizeof( Cell_E ), cudaMemcpyDeviceToHost ) );

    // pass result back
    args.ct = ct[0];
    //   args.hitcells_ct_h=hitcells_ct ;


  }

} // namespace CaloGpuGeneral_omp
