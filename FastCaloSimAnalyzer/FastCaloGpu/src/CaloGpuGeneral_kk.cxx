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

#define DEFAULT_BLOCK_SIZE 256

using namespace CaloGpuGeneral_fnc;

static std::once_flag calledGetEnv{};
static int            BLOCK_SIZE{DEFAULT_BLOCK_SIZE};

static int count {0};

static CaloGpuGeneral::KernelTime timing;

namespace CaloGpuGeneral_kk {

  void Rand4Hits_finish( void* /*rd4h*/ ) {

    // size_t free, total;
    // gpuQ( cudaMemGetInfo( &free, &total ) );
    // std::cout << "GPU memory used(MB): " << ( total - free ) / 1000000
    //           << "  bm table allocate size(MB), used:  "
    //           << DEV_BigMem::bm_ptr->size() / 1000000 << ", "
    //           << DEV_BigMem::bm_ptr->used() / 1000000
    //           << std::endl;
    // //    if ( (Rand4Hits*)rd4h ) delete (Rand4Hits*)rd4h;

    if (timing.count > 0) {
      std::cout << "kernel timing\n";
      std::cout << timing;
      // std::cout << "\n\n\n";
      // timing.printAll();
    } else {
      std::cout << "no kernel timing available" << std::endl;
    }
    
  }

  /* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

  
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

  /* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

  
  KOKKOS_INLINE_FUNCTION void rnd_to_fct2d( float& valuex, float& valuey, float rnd0, float rnd1, const FH2D_v& fh2d_v ) {

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

  KOKKOS_INLINE_FUNCTION void HistoLateralShapeParametrization_g_d( const HitParams hp, Hit& hit,
                                                                    unsigned long tid,
                                                                    const Sim_Args &args,
                                                                    const FH2D_v& fh2d_v,
                                                                    Kokkos::View<float*> rand_v )
  
  {

    float charge = hp.charge;

    // int cs=args.charge;
    float center_eta = hit.center_eta();
    float center_phi = hit.center_phi();
    float center_r   = hit.center_r();
    float center_z   = hit.center_z();

    float alpha, r, rnd1, rnd2;
    rnd1 = rand_v( tid );
    rnd2 = rand_v( tid + args.nhits );

    if ( hp.is_phi_symmetric ) {
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
    
    hit.m_eta_x = center_eta + delta_eta;
    
    hit.setEtaPhiZE( center_eta + delta_eta, center_phi + delta_phi, center_z, hit.E() );

    //    printf ("%f %f %f %f %f %f\n", center_eta, delta_eta, center_phi, delta_phi, center_z, hit.E());

    
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

  KOKKOS_INLINE_FUNCTION void HitCellMappingWiggle_g_d( HitParams hp, Hit& hit, unsigned long t,
                                                        const Sim_Args& args, const FHs_v& fhs_v ) {

    
    int                  nhist        = fhs_v.nhist();
    Kokkos::View<float*> bin_low_edge = fhs_v.low_edge;

    float eta = fabs( hit.eta() );
    if ( eta < bin_low_edge( 0 ) || eta > bin_low_edge( nhist ) ) { HitCellMapping_g_d( hp, hit, t, args ); }

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

  }

  /* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
  /* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

  void simulate_clean( Sim_Args args ) {
    Kokkos::View<float*> cellE_v( args.cells_energy, args.ncells*args.nsims );
    Kokkos::View<int*>   hitcells_ct_v( args.ct, args.nsims );
    Kokkos::deep_copy( cellE_v, 0. );
    Kokkos::deep_copy( hitcells_ct_v, 0 );
    Kokkos::fence();
  }

  /* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
  void simulate_hits_de( const Sim_Args args ) {

    Rand4Hits* rd4h = (Rand4Hits*)args.rd4h;

    auto                 maxhits = rd4h->get_t_a_hits();
    Kokkos::View<float*> rand_v( args.rand, 3 * maxhits );

    Kokkos::View<FHs_v*, Kokkos::HostSpace> f1v ("FHS_v",args.nbins);
    Kokkos::View<FH2D_v*, Kokkos::HostSpace> f2v ("FH2D_v", args.nbins);
    for (int i=0; i<args.nbins; ++i) {
      f2v(i) = *(args.hitparams_h[i].f2d_v);
      if (args.hitparams_h[i].cmw) {
        f1v(i) = *(args.hitparams_h[i].f1d_v);
      }
    }
    Kokkos::View<FHs_v*> f1d("FHS_v_d",args.nbins);
    Kokkos::View<FH2D_v*> f2d("FH2D_v_d",args.nbins);
    Kokkos::deep_copy(f1d,f1v);
    Kokkos::deep_copy(f2d,f2v);
    
    Kokkos::parallel_for(
        args.nhits, KOKKOS_LAMBDA( long tid ) {

          Hit hit;
          int bin = find_index_long( args.simbins, args.nbins, tid );
          HitParams hp = args.hitparams[bin];          
          hit.E() = hp.E;
          
          CenterPositionCalculation_g_d( hp, hit, tid, args );
          
          HistoLateralShapeParametrization_g_d( hp, hit, tid, args, f2d(bin), rand_v );
          if ( hp.cmw ) HitCellMappingWiggle_g_d( hp, hit, tid, args, f1d(bin) );
          
          HitCellMapping_g_d( hp, hit, tid, args );
        } );

    Kokkos::fence();

  }

  /* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
  void simulate_hits_ct( const Sim_Args args ) {

    unsigned long         ncells = args.ncells;
    unsigned long         nsims  = args.nsims;
    
    Kokkos::View<float*>  cellE_v( args.cells_energy, args.ncells*args.nsims );
    Kokkos::View<int*>    hitcells_ct_v( args.ct, args.nsims );
    Kokkos::View<Cell_E*> hitcells_E_v( args.hitcells_E, args.maxhitct*args.nsims );

    Kokkos::parallel_for(
        ncells*nsims, KOKKOS_LAMBDA( unsigned long tid ) {
          if ( tid < ncells * nsims ) {
            if ( cellE_v( tid ) > 0 ) {
              int           sim    = tid / ncells;
              unsigned long cellid = tid % ncells;
              
              //              printf("%ld %d %ld %f\n",tid,sim,cellid,(float)cellE_v(tid));

              unsigned int ct = Kokkos::atomic_fetch_add( &hitcells_ct_v(sim), 1 );
              Cell_E       ce;
              ce.cellid = cellid;
              ce.energy = cellE_v( tid );

              hitcells_E_v( ct + sim * MAXHITCT ) = ce;
            }
          }
        } );

    Kokkos::fence();
  }
  
  /* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
  void simulate_hits_gr( Sim_Args& args ) {

    // clean workspace
    auto t0   = std::chrono::system_clock::now();
    simulate_clean( args );

    // main simulation
    auto t1     = std::chrono::system_clock::now();
    simulate_hits_de( args );

    // stream compaction
    auto t2 = std::chrono::system_clock::now();
    simulate_hits_ct( args );

    // copy back to host
    auto t3 = std::chrono::system_clock::now();
    Kokkos::View<int*>             ctd( args.ct, args.nsims );
    Kokkos::View<int*, Kokkos::HostSpace> cth( args.ct_h, args.nsims );
    Kokkos::deep_copy( cth, ctd );

    Kokkos::View<Cell_E*>                    cev(  args.hitcells_E,   MAXHITCT*args.nsims );
    Kokkos::View<Cell_E*, Kokkos::HostSpace> cevh( args.hitcells_E_h, MAXHITCT*args.nsims );
    Kokkos::deep_copy( cevh, cev );

    auto t4 = std::chrono::system_clock::now();
    
#ifdef DUMP_HITCELLS
    std::cout << "========= Listing HitCells =========\n";
    std::cout << "nsim: " << args.nsims << "\n";
    for (int isim=0; isim<args.nsims; ++isim) {
      std::cout << "  nhit: " << args.ct_h[isim] << "\n";
      std::map<unsigned int,float> cm;
      for (int ihit=0; ihit<args.ct_h[isim]; ++ihit) {
        cm[args.hitcells_E_h[ihit+isim*MAXHITCT].cellid] = args.hitcells_E_h[ihit+isim*MAXHITCT].energy;
      }

      int i=0;
      for (auto &em: cm) {
        std::cout << "   " << isim << " " << i++ << "  cell: " << em.first << "  " << em.second << std::endl;
      }
    }
    std::cout << "====================================\n";
#endif
    
    timing.add( t1 - t0, t2 - t1, t3 - t2, t4 - t3 );
      
  }


  /* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

  void load_hitsim_params( void* rd4h, HitParams* hp, long* simbins, int bins ) {

  if ( !(Rand4Hits*)rd4h ) {
    std::cout << "Error load hit simulation params ! ";
    exit( 2 );
  }

  Kokkos::View<HitParams*, Kokkos::HostSpace> hpv_h( hp, bins );
  Kokkos::deep_copy(Kokkos::subview( ( *((Rand4Hits*)rd4h )->get_hitparams_v()),
                                     Kokkos::make_pair(0, bins)), hpv_h );

  Kokkos::View<long*, Kokkos::HostSpace> sbv_h( simbins, bins );
  Kokkos::deep_copy(Kokkos::subview( ( *((Rand4Hits*)rd4h )->get_simbins_v()),
                                     Kokkos::make_pair(0, bins)), sbv_h );
  
}

  

} // namespace CaloGpuGeneral_kk
