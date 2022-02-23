/*
  Copyright (C) 2002-2021 CERN for the benefit of the ATLAS collaboration
*/

#include "GeoRegion.h"
#include "GeoGpu_structs.h"
#include "Hit.h"
#include "Rand4Hits.h"
#include "Args.h"

#if defined( USE_STDPAR )
#  define __DEVICE__
#  define __HOST__
#else
#  define __DEVICE__ __device__
#  define __HOST__ __host__
#endif

namespace CaloGpuGeneral_fnc {
  __DEVICE__ long long getDDE( GeoGpu* geo, int sampling, float eta, float phi ) {
    float* distance = 0;
    int*   steps    = 0;

    int              MAX_SAMPLING = geo->max_sample;
    Rg_Sample_Index* SampleIdx    = geo->sample_index;
    GeoRegion*       regions_g    = geo->regions;

    if ( sampling < 0 ) return -1;
    if ( sampling >= MAX_SAMPLING ) return -1;

    int sample_size  = SampleIdx[sampling].size;
    int sample_index = SampleIdx[sampling].index;

    GeoRegion* gr = (GeoRegion*)regions_g;
    if ( sample_size == 0 ) return -1;
    float     dist;
    long long bestDDE = -1;
    if ( !distance ) distance = &dist;
    *distance = +10000000;
    int intsteps;
    int beststeps;
    if ( steps )
      beststeps = ( *steps );
    else
      beststeps = 0;

    if ( sampling < 21 ) {
      for ( int skip_range_check = 0; skip_range_check <= 1; ++skip_range_check ) {
        for ( unsigned int j = sample_index; j < sample_index + sample_size; ++j ) {
          if ( !skip_range_check ) {
            if ( eta < gr[j].mineta() ) continue;
            if ( eta > gr[j].maxeta() ) continue;
          }
          if ( steps )
            intsteps = ( *steps );
          else
            intsteps = 0;
          float     newdist;
          long long newDDE = gr[j].getDDE( eta, phi, &newdist, &intsteps );
          if ( newdist < *distance ) {
            bestDDE   = newDDE;
            *distance = newdist;
            if ( steps ) beststeps = intsteps;
            if ( newdist < -0.1 ) break; // stop, we are well within the hit cell
          }
        }
        if ( bestDDE >= 0 ) break;
      }
    } else {
      return -3;
    }
    if ( steps ) *steps = beststeps;

    return bestDDE;
  }

  /* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

  __DEVICE__ int find_index_f( float* array, int size, float value ) {
    // fist index (from 0)  have element value > value
    // array[i] > value ; array[i-1] <= value
    // std::upbund( )
    int low     = 0;
    int high    = size - 1;
    int m_index = ( high - low ) / 2;
    while ( m_index != high ) {
      if ( value < array[m_index] )
        high = m_index;
      else
        low = m_index + 1;
      m_index = ( high + low + 1 ) / 2;
    }
    return m_index;
  }

  /* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

  __DEVICE__ int find_index_uint32( uint32_t* array, int size, uint32_t value ) {
    // fist index i  have element value > value
    // array[i] > value ; array[i-1] <= value
    int low     = 0;
    int high    = size - 1;
    int m_index = ( high - low ) / 2;
    while ( m_index != high ) {
      if ( value < array[m_index] )
        high = m_index;
      else
        low = m_index + 1;
      m_index = ( high + low + 1 ) / 2;
    }
    return m_index;
  }

  /* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

  __DEVICE__ int find_index_long( long* array, int size, long value ) {
    // find the first index of element which has vaule > value
    int low     = 0;
    int high    = size - 1;
    int m_index = ( high - low ) / 2;
    while ( high != low ) {
      if ( value > array[m_index] )
        low = m_index + 1;
      else if ( value == array[m_index] ) {
        return m_index + 1;
        // return min(m_index + 1, size-1)   ;
      } else
        high = m_index;
      m_index = ( high - low ) / 2 + low;
    }
    return m_index;
  }

  /* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

  __DEVICE__ void rnd_to_fct2d( float& valuex, float& valuey, float rnd0, float rnd1, FH2D* hf2d ) {

    int    nbinsx        = ( *hf2d ).nbinsx;
    int    nbinsy        = ( *hf2d ).nbinsy;
    float* HistoContents = ( *hf2d ).h_contents;
    float* HistoBorders  = ( *hf2d ).h_bordersx;
    float* HistoBordersy = ( *hf2d ).h_bordersy;

    int ibin = find_index_f( HistoContents, nbinsx * nbinsy, rnd0 );

    int biny = ibin / nbinsx;
    int binx = ibin - nbinsx * biny;

    float basecont = 0;
    if ( ibin > 0 ) basecont = HistoContents[ibin - 1];

    float dcont = HistoContents[ibin] - basecont;
    if ( dcont > 0 ) {
      valuex = HistoBorders[binx] + ( HistoBorders[binx + 1] - HistoBorders[binx] ) * ( rnd0 - basecont ) / dcont;
    } else {
      valuex = HistoBorders[binx] + ( HistoBorders[binx + 1] - HistoBorders[binx] ) / 2;
    }
    valuey = HistoBordersy[biny] + ( HistoBordersy[biny + 1] - HistoBordersy[biny] ) * rnd1;
  }

  /* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

  __DEVICE__ float rnd_to_fct1d( float rnd, uint32_t* contents, float* borders, int nbins, uint32_t s_MaxValue,
                                 unsigned long t ) {

    uint32_t int_rnd = s_MaxValue * rnd;
    int      ibin    = find_index_uint32( contents, nbins, int_rnd );

    int binx = ibin;

    uint32_t basecont = 0;
    if ( ibin > 0 ) basecont = contents[ibin - 1];

    uint32_t dcont = contents[ibin] - basecont;
    if ( dcont > 0 ) {
      return borders[binx] + ( ( borders[binx + 1] - borders[binx] ) * ( int_rnd - basecont ) ) / dcont;
    } else {
      return borders[binx] + ( borders[binx + 1] - borders[binx] ) / 2;
    }
  }

  /* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

  __DEVICE__ void CenterPositionCalculation_g_d( const HitParams hp, Hit& hit, long tid, const Sim_Args args ) {
    
    hit.setCenter_r( ( 1. - hp.extrapWeight ) * hp.extrapol_r_ent + hp.extrapWeight * hp.extrapol_r_ext );
    hit.setCenter_z( ( 1. - hp.extrapWeight ) * hp.extrapol_z_ent + hp.extrapWeight * hp.extrapol_z_ext );
    hit.setCenter_eta( ( 1. - hp.extrapWeight ) * hp.extrapol_eta_ent + hp.extrapWeight * hp.extrapol_eta_ext );
    hit.setCenter_phi( ( 1. - hp.extrapWeight ) * hp.extrapol_phi_ent + hp.extrapWeight * hp.extrapol_phi_ext );
    
}

  /* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

  __DEVICE__ void HistoLateralShapeParametrization_g_d( const HitParams hp, Hit& hit, unsigned long t, Sim_Args args ) {

    float charge = hp.charge;

    float center_eta = hit.center_eta();
    float center_phi = hit.center_phi();
    float center_r   = hit.center_r();
    float center_z   = hit.center_z();

    float alpha, r, rnd1, rnd2;
    rnd1 = args.rand[t];
    rnd2 = args.rand[t + args.nhits];

    if ( hp.is_phi_symmetric ) {
      if ( rnd2 >= 0.5 ) { // Fill negative phi half of shape
        rnd2 -= 0.5;
        rnd2 *= 2;
        rnd_to_fct2d( alpha, r, rnd1, rnd2, hp.f2d );
        alpha = -alpha;
      } else { // Fill positive phi half of shape
        rnd2 *= 2;
        rnd_to_fct2d( alpha, r, rnd1, rnd2, hp.f2d );
      }
    } else {
      rnd_to_fct2d( alpha, r, rnd1, rnd2, hp.f2d );
    }

    float delta_eta_mm = r * cos( alpha );
    float delta_phi_mm = r * sin( alpha );

    // Particles with negative eta are expected to have the same shape as those with positive eta after transformation:
    // delta_eta --> -delta_eta
    if ( center_eta < 0. ) delta_eta_mm = -delta_eta_mm;
    // Particle with negative charge are expected to have the same shape as positively charged particles after
    // transformation: delta_phi --> -delta_phi
    if ( charge < 0. ) delta_phi_mm = -delta_phi_mm;

    float dist000    = sqrt( center_r * center_r + center_z * center_z );
    float eta_jakobi = abs( 2.0 * exp( -center_eta ) / ( 1.0 + exp( -2 * center_eta ) ) );

    float delta_eta = delta_eta_mm / eta_jakobi / dist000;
    float delta_phi = delta_phi_mm / center_r;

    hit.setEtaPhiZE( center_eta + delta_eta, center_phi + delta_phi, center_z, hit.E() );

  }

  /* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

  __DEVICE__ void HitCellMapping_g_d( HitParams hp, Hit& hit, unsigned long t, Sim_Args args ) {

    long long cellele = getDDE( args.geo, hp.cs, hit.eta(), hit.phi() );

    if ( cellele < 0 ) {
      printf( "HitCellMapping_g_d: cellele not found  eta: %f  phi: %f\n", hit.eta(), hit.phi() );
      return;
    }

#if defined( USE_STDPAR )
    //    printf("HCM_b: %lu %f %lld %lu\n", t, hit.E(), cellele, (int)args.cells_energy[cellele]);
#  ifdef _NVHPC_STDPAR_NONE
    args.cells_energy[cellele + args.ncells * hp.index] += hit.E();
#  else
    args.cells_energy[cellele + args.ncells * hp.index] += int( CELL_ENE_FAC * hit.E() );
#  endif
    //    printf("HCM_b: %lu %f %lld %lu\n", t, hit.E(), cellele, (int)args.cells_energy[cellele]);

#else
    atomicAdd( &args.cells_energy[cellele + args.ncells * hp.index], hit.E() );
#endif

    //    printf("HCM: %lu %f %lld %f\n", t, hit.E(), cellele, args.cells_energy[cellele]);
  }

  /* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

  __DEVICE__ void HitCellMappingWiggle_g_d( HitParams hp, Hit& hit, unsigned long t, Sim_Args args ) {

    FHs*   f1d          = hp.f1d;
    int    nhist        = ( *f1d ).nhist;
    float* bin_low_edge = ( *f1d ).low_edge;

    float eta = fabs( hit.eta() );
    if ( eta < bin_low_edge[0] || eta > bin_low_edge[nhist] ) { HitCellMapping_g_d( hp, hit, t, args ); }

    int bin = nhist;
    for ( int i = 0; i < nhist + 1; ++i ) {
      if ( bin_low_edge[i] > eta ) {
        bin = i;
        break;
      }
    }

    //  bin=find_index_f(bin_low_edge, nhist+1, eta ) ;

    bin -= 1;

    uint32_t* contents   = ( *f1d ).h_contents[bin];
    float*    borders    = ( *f1d ).h_borders[bin];
    int       h_size     = ( *f1d ).h_szs[bin];
    uint32_t  s_MaxValue = ( *f1d ).s_MaxValue;

    float rnd = args.rand[t + 2 * args.nhits];

    float wiggle = rnd_to_fct1d( rnd, contents, borders, h_size, s_MaxValue, t );

    //    printf("HCMW: %ld %f %f %f, %d, %d, %d\n",t,hit.phi(), wiggle, rnd, bin, h_size, s_MaxValue);
    float hit_phi_shifted = hit.phi() + wiggle;
    hit.phi()             = Phi_mpi_pi( hit_phi_shifted );

    //    HitCellMapping_g_d( hp, hit, t, args );
  }

  /* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

  __HOST__ int highestPowerof2( unsigned int n ) {
    // Invalid input
    if ( n < 1 ) return 0;

    int res = 1;

    // Try all powers starting from 2^1
    for ( unsigned int i = 0; i < 8 * sizeof( unsigned int ); i++ ) {
      unsigned int curr = 1 << i;

      // If current power is more than n, break
      if ( curr > n ) break;

      res = curr;
    }

    return res;
  }

} // namespace CaloGpuGeneral_fnc
