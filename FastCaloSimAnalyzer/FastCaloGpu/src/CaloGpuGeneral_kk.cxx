/*
  Copyright (C) 2002-2021 CERN for the benefit of the ATLAS collaboration
*/

#include "CaloGpuGeneral_kk.h"
#include "CaloGpuGeneral_cu.h"
#include "Rand4Hits.h"
#include "FH_views.h"
#include "Hit.h"

#include <Kokkos_Core.hpp>
#include <Kokkos_Random.hpp>

static CaloGpuGeneral::KernelTime timing;

using namespace CaloGpuGeneral_fnc;

namespace CaloGpuGeneral_kk {

  KOKKOS_INLINE_FUNCTION int find_index_f( Kokkos::View<float*> array, int size, float value ) {
    // fist index (from 0)  have element value > value
    // array[i] > value ; array[i-1] <= value
    // std::upbund( )

    int low     = 0;
    int high    = size - 1;
    int m_index = ( high - low ) / 2;
    while ( high != low ) {
      if ( value >= array( m_index ) )
        low = m_index + 1;
      else
        high = m_index;
      m_index = ( high + low ) / 2;
    }
    return m_index;
  }

  KOKKOS_INLINE_FUNCTION void rnd_to_fct2d( float& valuex, float& valuey, float rnd0, float rnd1, FH2D_v fh2d_v ) {

    int nbinsx = fh2d_v.nbinsx();
    int nbinsy = fh2d_v.nbinsy();
    int ibin   = find_index_f( fh2d_v.contents, nbinsx * nbinsy, rnd0 );

    int biny = ibin / nbinsx;
    int binx = ibin - nbinsx * biny;

    float basecont = 0;
    if ( ibin > 0 ) basecont = fh2d_v.contents( ibin - 1 );

    float dcont = fh2d_v.contents( ibin ) - basecont;
    if ( dcont > 0 ) {
      valuex = fh2d_v.bordersx( binx ) +
               ( fh2d_v.bordersx( binx + 1 ) - fh2d_v.bordersx( binx ) ) * ( rnd0 - basecont ) / dcont;
    } else {
      valuex = fh2d_v.bordersx( binx ) + ( fh2d_v.bordersx( binx + 1 ) - fh2d_v.bordersx( binx ) ) / 2;
    }
    valuey = fh2d_v.bordersy( biny ) + ( fh2d_v.bordersy( biny + 1 ) - fh2d_v.bordersy( biny ) ) * rnd1;
  }

  /* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

  KOKKOS_INLINE_FUNCTION void HistoLateralShapeParametrization_d( Hit& hit, unsigned long t, Chain0_Args args,
                                                                  FH2D_v fh2d_v, Kokkos::View<float*> rand_v ) {

    // int     pdgId    = args.pdgId;
    float charge = args.charge;

    // int cs=args.charge;
    float center_eta = hit.center_eta();
    float center_phi = hit.center_phi();
    float center_r   = hit.center_r();
    float center_z   = hit.center_z();

    float alpha, r, rnd1, rnd2;
    rnd1 = rand_v( t );
    rnd2 = rand_v( t + args.nhits );

    if ( args.is_phi_symmetric ) {
      if ( rnd2 >= 0.5 ) { // Fill negative phi half of shape
        rnd2 -= 0.5;
        rnd2 *= 2;
        rnd_to_fct2d( alpha, r, rnd1, rnd2, fh2d_v );
        alpha = -alpha;
      } else { // Fill positive phi half of shape
        rnd2 *= 2;
        rnd_to_fct2d( alpha, r, rnd1, rnd2, fh2d_v );
      }
    } else {
      rnd_to_fct2d( alpha, r, rnd1, rnd2, fh2d_v );
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
  KOKKOS_INLINE_FUNCTION int find_index_uint32( Kokkos::View<uint32_t*> array, int size, uint32_t value ) {
    // find the first index of element which has vaule > value
    int low     = 0;
    int high    = size - 1;
    int m_index = ( high - low ) / 2;
    while ( high != low ) {
      if ( value > array( m_index ) )
        low = m_index + 1;
      else if ( value == array( m_index ) ) {
        return m_index + 1;
      } else
        high = m_index;
      m_index = ( high - low ) / 2 + low;
    }
    return m_index;
  }

  /* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

  KOKKOS_INLINE_FUNCTION float rnd_to_fct1d( float rnd, Kokkos::View<uint32_t*> contents, Kokkos::View<float*> borders,
                                             int nbins, uint32_t s_MaxValue ) {

    uint32_t int_rnd = s_MaxValue * rnd;
    int      ibin    = find_index_uint32( contents, nbins, int_rnd );

    int binx = ibin;

    uint32_t basecont = 0;
    if ( ibin > 0 ) basecont = contents( ibin - 1 );

    uint32_t dcont = contents( ibin ) - basecont;
    if ( dcont > 0 ) {
      return borders( binx ) + ( ( borders( binx + 1 ) - borders( binx ) ) * ( int_rnd - basecont ) ) / dcont;
    } else {
      return borders( binx ) + ( borders( binx + 1 ) - borders( binx ) ) / 2;
    }
  }

  /* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

  KOKKOS_INLINE_FUNCTION void HitCellMappingWiggle_d( Hit& hit, Chain0_Args args, unsigned long t, FHs_v fhs_v ) {

    int                  nhist        = fhs_v.nhist();
    Kokkos::View<float*> bin_low_edge = fhs_v.low_edge;

    float eta = fabs( hit.eta() );
    if ( eta < bin_low_edge( 0 ) || eta > bin_low_edge( nhist ) ) { HitCellMapping_d( hit, t, args ); }

    int bin = nhist;
    for ( int i = 0; i < nhist + 1; ++i ) {
      if ( bin_low_edge( i ) > eta ) {
        bin = i;
        break;
      }
    }

    bin -= 1;

    Kokkos::View<uint32_t*> contents   = fhs_v.d_contents1D;
    Kokkos::View<float*>    borders    = fhs_v.d_borders1D;
    int                     h_size     = fhs_v.h_szs( bin );
    uint32_t                s_MaxValue = fhs_v.s_MaxValue();

    float rnd = args.rand[t + 2 * args.nhits];

    float wiggle = rnd_to_fct1d( rnd, contents, borders, h_size, s_MaxValue );

    float hit_phi_shifted = hit.phi() + wiggle;
    hit.phi()             = Phi_mpi_pi( hit_phi_shifted );

    HitCellMapping_d( hit, t, args );
  }

  /* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
  /* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

  void simulate_clean( Chain0_Args args ) {
    Kokkos::View<float*> cellE_v( args.cells_energy, args.ncells );
    Kokkos::View<int*>   hitcells_ct_v( args.hitcells_ct, 1 );
    Kokkos::deep_copy( cellE_v, 0. );
    Kokkos::deep_copy( hitcells_ct_v, 0. );
  }

  /* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
  void simulate_A( float E, int nhits, Chain0_Args args ) {

    Rand4Hits* rd4h = (Rand4Hits*)args.rd4h;

    auto                 maxhits = rd4h->get_t_a_hits();
    Kokkos::View<float*> rand_v( args.rand, 3 * maxhits );
    FH2D_v               fh2d_v( args );
    FHs_v                fhs_v( args );

    Kokkos::parallel_for(
        nhits, KOKKOS_LAMBDA( int tid ) {
          Hit hit;
          hit.E() = E;

          CenterPositionCalculation_d( hit, args );
          HistoLateralShapeParametrization_d( hit, tid, args, fh2d_v, rand_v );
          HitCellMappingWiggle_d( hit, args, tid, fhs_v );
        } );
  }

  /* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
  void simulate_ct( Chain0_Args args ) {

    int                   ncells = args.ncells;
    Kokkos::View<float*>  cellE_v( args.cells_energy, args.ncells );
    Kokkos::View<int>     hitcells_ct_v( args.hitcells_ct );
    Kokkos::View<Cell_E*> hitcells_E_v( args.hitcells_E, args.maxhitct );

    Kokkos::parallel_for(
        ncells, KOKKOS_LAMBDA( int tid ) {
          if ( cellE_v( tid ) > 0 ) {
            unsigned int ct = Kokkos::atomic_fetch_add( &hitcells_ct_v(), 1 );
            Cell_E       ce;
            ce.cellid = tid;
            ce.energy = cellE_v( tid );

            hitcells_E_v( ct ) = ce;
          }
        } );
  }

  /* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
  void simulate_hits( float E, int nhits, Chain0_Args& args ) {

    auto t0 = std::chrono::system_clock::now();
    simulate_clean( args );
    Kokkos::fence();
 
    auto t1 = std::chrono::system_clock::now();
    simulate_A( E, nhits, args );
    Kokkos::fence();
 
    auto t2 = std::chrono::system_clock::now();
    simulate_ct( args );
    Kokkos::fence();
 
    auto t3 = std::chrono::system_clock::now();
    Kokkos::View<int>             ctd( args.hitcells_ct );
    Kokkos::View<int>::HostMirror cth = Kokkos::create_mirror_view( ctd );
    Kokkos::deep_copy( cth, ctd );

    Kokkos::View<Cell_E*>                    cev( args.hitcells_E, cth() );
    Kokkos::View<Cell_E*, Kokkos::HostSpace> cevh( args.hitcells_E_h, cth() );
    Kokkos::deep_copy( cevh, cev );
    Kokkos::fence();
    
    // pass result back
    args.ct = cth();
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
    
    timing.add( t1 - t0, t2 - t1, t3 - t2, t4 - t3 );

  }

  /* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
  void Rand4Hits_finish( void* rd4h ) {
    if ( (Rand4Hits*)rd4h ) delete (Rand4Hits*)rd4h;

    std::cout << timing;
  }


} // namespace CaloGpuGeneral_kk
