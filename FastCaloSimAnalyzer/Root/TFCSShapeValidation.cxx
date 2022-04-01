/*
  Copyright (C) 2002-2017 CERN for the benefit of the ATLAS collaboration
 */

#include "FastCaloSimAnalyzer/TFCSShapeValidation.h"

#include "TString.h"
#include "TMath.h"
#include "TFile.h"
#include "TH1.h"
#include "TH2.h"
#include "TProfile2D.h"
#include "TCanvas.h"

#include "TChain.h"

#include <iostream>
#include <tuple>
#include <map>
#include <algorithm>
#include <fstream>

#include "CLHEP/Random/TRandomEngine.h"

#include "ISF_FastCaloSimEvent/TFCSSimulationState.h"
#include "ISF_FastCaloSimEvent/TFCSTruthState.h"
#include "ISF_FastCaloSimEvent/TFCSExtrapolationState.h"

#include "TFCSSampleDiscovery.h"

#include <chrono>
#include <typeinfo>

#if defined USE_GPU || defined USE_OMPGPU
#  include "FastCaloGpu/FastCaloGpu/GeoLoadGpu.h"
#  include "FastCaloGpu/FastCaloGpu/CaloGpuGeneral.h"
#  include <omp.h>
#endif

#ifdef USE_KOKKOS
#  include <Kokkos_Core.hpp>
#endif

std::chrono::duration<double> TFCSShapeValidation::time_g1;
std::chrono::duration<double> TFCSShapeValidation::time_g2;
std::chrono::duration<double> TFCSShapeValidation::time_o1;
std::chrono::duration<double> TFCSShapeValidation::time_o2;
std::chrono::duration<double> TFCSShapeValidation::time_h;
std::chrono::duration<double> TFCSShapeValidation::time_nhits;
std::chrono::duration<double> TFCSShapeValidation::time_mchain;
std::chrono::duration<double> TFCSShapeValidation::time_hitsim;

TFCSShapeValidation::TFCSShapeValidation( long seed ) {
  m_debug      = 0;
  m_geo        = 0;
  m_nprint     = -1;
  m_firstevent = 0;

  m_randEngine = new CLHEP::TRandomEngine();
  m_randEngine->setSeed( seed );

#if defined USE_GPU || defined USE_OMPGPU
  m_gl   = 0;
  m_rd4h = CaloGpuGeneral::Rand4Hits_init( MAXHITS, MAXBINS, seed, true );
#endif
}

TFCSShapeValidation::TFCSShapeValidation( TChain* chain, int layer, long seed ) {
  m_debug      = 0;
  m_chain      = chain;
  m_output     = "";
  m_layer      = layer;
  m_geo        = 0;
  m_nprint     = -1;
  m_firstevent = 0;

  m_randEngine = new CLHEP::TRandomEngine();
  auto                          t_bgn = std::chrono::system_clock::now();
  m_randEngine->setSeed( seed );
  auto                          t_end = std::chrono::system_clock::now();
  std::chrono::duration<double> diff1 = t_end - t_bgn;
  std::cout << "Time to seed rands on CPU: " << diff1.count() << " s" << std::endl;
#if defined USE_GPU
//#if defined USE_GPU || defined USE_OMPGPU
  auto                            t0 = std::chrono::system_clock::now();
  m_gl                               = 0;
  m_rd4h                             = CaloGpuGeneral::Rand4Hits_init( MAXHITS, MAXBINS, seed, true );
  auto                            t1 = std::chrono::system_clock::now();
  std::chrono::duration<double> diff = t1 - t0;
  std::cout << "Time of Rand4Hit_init: " << diff.count() << " s" << std::endl;
#endif
}

TFCSShapeValidation::~TFCSShapeValidation() {}

void TFCSShapeValidation::LoadGeo() {
  if ( m_geo ) return;
  m_geo = new CaloGeometryFromFile();

  // load geometry files
  m_geo->LoadGeometryFromFile( TFCSSampleDiscovery::geometryName(), TFCSSampleDiscovery::geometryTree(),
                               TFCSSampleDiscovery::geometryMap() );
  m_geo->LoadFCalGeometryFromFiles( TFCSSampleDiscovery::geometryNameFCal() );
}

void TFCSShapeValidation::LoopEvents( int pcabin = -1 ) {

  auto start = std::chrono::system_clock::now();

  LoadGeo();

  auto                          t_01 = std::chrono::system_clock::now();
  std::chrono::duration<double> diff;
  time_o1 = std::chrono::duration<double, std::ratio<1>>::zero();
  time_o2 = std::chrono::duration<double, std::ratio<1>>::zero();
  time_g1 = std::chrono::duration<double, std::ratio<1>>::zero();
  time_g2 = std::chrono::duration<double, std::ratio<1>>::zero();
  time_h  = std::chrono::duration<double, std::ratio<1>>::zero();
  time_nhits = std::chrono::duration<double, std::ratio<1>>::zero();
  time_mchain = std::chrono::duration<double, std::ratio<1>>::zero();
  time_hitsim = std::chrono::duration<double, std::ratio<1>>::zero();

  std::chrono::duration<double> t_c[5] = {std::chrono::duration<double, std::ratio<1>>::zero()};
  std::chrono::duration<double> t_c1[5] = {std::chrono::duration<double, std::ratio<1>>::zero()};
  std::chrono::duration<double> t_c2[5] = {std::chrono::duration<double, std::ratio<1>>::zero()};
  std::chrono::duration<double> t_bc   = std::chrono::duration<double, std::ratio<1>>::zero();
  std::chrono::duration<double> t_io   = std::chrono::duration<double, std::ratio<1>>::zero();

#ifdef USE_GPU
  GeoLg();

  if ( m_gl->LoadGpu() ) std::cout << "GPU Geometry loaded!!!" << std::endl;

  // m_debug=1 ;
  auto t1 = std::chrono::system_clock::now();

  if ( 0 ) {
    std::cout << "Geo size: " << m_geo->get_cells()->size() << std::endl;
    std::cout << "Geo region size: ";
    for ( int isample = 0; isample < 24; isample++ ) { std::cout << m_geo->get_n_regions( isample ) << " "; }
    std::cout << std::endl;

    unsigned long t_cells = 0;
    for ( int isample = 0; isample < 24; isample++ ) {
      std::cout << "Sample: " << isample << std::endl;
      int sample_tot = 0;
      int rgs        = m_geo->get_n_regions( isample );
      for ( int irg = 0; irg < rgs; irg++ ) {
        std::cout << " region: " << irg << " cells: " << m_geo->get_region_size( isample, irg ) << std::endl;
        sample_tot += m_geo->get_region_size( isample, irg );
        t_cells += m_geo->get_region_size( isample, irg );
        int neta = m_geo->get_region( isample, irg )->cell_grid_eta();
        int nphi = m_geo->get_region( isample, irg )->cell_grid_phi();
        std::cout << "     Cell Grid neta,nphi :" << neta << "  " << nphi << std::endl;
      }
      std::cout << "Total cells for sample " << isample << " is " << sample_tot << std::endl;
    }
    std::cout << "Total cells for all regions and samples: " << t_cells << std::endl;
  }
#elif defined USE_OMPGPU
  ////////////////////////////////
  // Load geometry on GPU using OMP
  ////////////////////////////////
  std::cout << "Loading geometry on GPU using OMP..." << std::endl;
  auto               t_bgn_loadgeogpu = std::chrono::system_clock::now();
  GeoLg();
  if ( m_gl->LoadGpu() ) std::cout << "Success!!!" << std::endl;
  auto               t_end_loadgeogpu = std::chrono::system_clock::now();
  std::chrono::duration<double> diff2 = t_end_loadgeogpu - t_bgn_loadgeogpu;
  std::cout << "Time to load geo on GPU: " << diff2.count() << " s" << std::endl;
#endif

  int nentries = m_nentries;
  int layer    = m_layer;
  std::cout << "TFCSShapeValidation::LoopEvents(): Running on layer = " << layer << ", pcabin = " << pcabin
            << std::endl;

  InitInputTree( m_chain, layer );
#ifdef USE_GPU
  auto t_02 = std::chrono::system_clock::now();
#endif

  ///////////////////////////////////
  //// Initialize truth, extraplolation and all validation structures
  ///////////////////////////////////
  m_truthTLV.resize( nentries );
  m_extrapol.resize( nentries );
#ifdef USE_GPU
  auto t_03 = std::chrono::system_clock::now();
#endif

  for ( auto& validation : m_validations ) {
    std::cout << "========================================================" << std::endl;
    if ( m_debug >= 1 ) validation.basesim()->setLevel( MSG::DEBUG, true );
    validation.basesim()->set_geometry( m_geo );
#ifdef FCS_DEBUG
    validation.basesim()->Print();
#endif
    validation.simul().reserve( nentries );
    std::cout << "========================================================" << std::endl << std::endl;
  }

  ///////////////////////////////////
  //// Event loop
  ///////////////////////////////////
  if ( m_nprint < 0 ) {
    m_nprint = 250;
    if ( nentries < 5000 ) m_nprint = 100;
    if ( nentries < 1000 ) m_nprint = 50;
    if ( nentries < 500 ) m_nprint = 20;
    if ( nentries < 100 ) m_nprint = 1;
  }

#ifdef USE_GPU
  TFCSSimulationState::EventStatus es = {-1, false, false};
#endif
  auto t2 = std::chrono::system_clock::now();

  ///////////////////////////////////
  //// Event loop
  ///////////////////////////////////
  for ( int ievent = m_firstevent; ievent < nentries; ievent++ ) {

#ifdef USE_GPU
    es.ievent = ievent;

    bool first  = ( ievent == m_firstevent ) ? true : false;
    es.is_first = first;

    bool last  = ( ievent == ( nentries - 1 ) ) ? true : false;
    es.is_last = last;

#endif

    if ( ievent % m_nprint == 0 ) std::cout << std::endl << "Event: " << ievent << std::endl;
    auto    t6         = std::chrono::system_clock::now();
    int64_t localEntry = m_chain->LoadTree( ievent );
    for ( TBranch* branch : m_branches ) { branch->GetEntry( localEntry ); }
    auto t7 = std::chrono::system_clock::now();
    t_io += t7 - t6;

    if ( m_debug >= 1 ) { std::cout << "Number of particles: " << m_truthPDGID->size() << std::endl; }

    size_t particles = m_truthPDGID->size();
    // std::cout << std::endl << "Event: " << ievent <<"Number of Particles: "<< particles << std::endl;
    for ( size_t p = 0; p < particles; p++ ) {

      auto t4 = std::chrono::system_clock::now();

      ///////////////////////////////////
      //// Initialize truth
      ///////////////////////////////////
      float px    = m_truthPx->at( p );
      float py    = m_truthPy->at( p );
      float pz    = m_truthPz->at( p );
      float E     = m_truthE->at( p );
      int   pdgid = m_truthPDGID->at( p );

      TFCSTruthState& truthTLV = m_truthTLV[ievent];
      truthTLV.SetPxPyPzE( px, py, pz, E );
      truthTLV.set_pdgid( pdgid );

      ///////////////////////////////////
      //// OLD, to be removed: should run over all pca bins
      ///////////////////////////////////

      if ( m_debug >= 1 ) {
        std::cout << std::endl << "Event: " << ievent;
        std::cout << " pca = " << pca() << " m_pca=" << m_pca << " " << std::endl;
        truthTLV.Print();
      }

      ///////////////////////////////////
      //// Initialize truth extrapolation to each calo layer
      ///////////////////////////////////
      TFCSExtrapolationState& extrapol = m_extrapol[ievent];
      extrapol.clear();

      float TTC_eta, TTC_phi, TTC_r, TTC_z;

      if ( !m_isNewSample ) {
        TTC_eta = ( *m_truthCollection )[0].TTC_entrance_eta[0];
        TTC_phi = ( *m_truthCollection )[0].TTC_entrance_phi[0];
        TTC_r   = ( *m_truthCollection )[0].TTC_entrance_r[0];
        TTC_z   = ( *m_truthCollection )[0].TTC_entrance_z[0];

        std::cout << std::endl << " TTC size: " << ( *m_truthCollection )[0].TTC_entrance_eta.size() << std::endl;

        for ( int i = 0; i < CaloCell_ID_FCS::MaxSample; ++i ) {
          if ( m_total_layer_cell_energy[i] == 0 ) continue;
          extrapol.set_OK( i, TFCSExtrapolationState::SUBPOS_ENT, true );
          extrapol.set_eta( i, TFCSExtrapolationState::SUBPOS_ENT, ( *m_truthCollection )[0].TTC_entrance_eta[i] );
          extrapol.set_phi( i, TFCSExtrapolationState::SUBPOS_ENT, ( *m_truthCollection )[0].TTC_entrance_phi[i] );
          extrapol.set_r( i, TFCSExtrapolationState::SUBPOS_ENT, ( *m_truthCollection )[0].TTC_entrance_r[i] );
          extrapol.set_z( i, TFCSExtrapolationState::SUBPOS_ENT, ( *m_truthCollection )[0].TTC_entrance_z[i] );

          extrapol.set_OK( i, TFCSExtrapolationState::SUBPOS_EXT, true );
          extrapol.set_eta( i, TFCSExtrapolationState::SUBPOS_EXT, ( *m_truthCollection )[0].TTC_back_eta[i] );
          extrapol.set_phi( i, TFCSExtrapolationState::SUBPOS_EXT, ( *m_truthCollection )[0].TTC_back_phi[i] );
          extrapol.set_r( i, TFCSExtrapolationState::SUBPOS_EXT, ( *m_truthCollection )[0].TTC_back_r[i] );
          extrapol.set_z( i, TFCSExtrapolationState::SUBPOS_EXT, ( *m_truthCollection )[0].TTC_back_z[i] );

          // extrapol.set_OK(i,TFCSExtrapolationState::SUBPOS_MID, true);
          // extrapol.set_eta(i,TFCSExtrapolationState::SUBPOS_MID, m_TTC_mid_eta->at(p).at(i));
          // extrapol.set_phi(i,TFCSExtrapolationState::SUBPOS_MID, m_TTC_mid_phi->at(p).at(i));
          // extrapol.set_r(i,TFCSExtrapolationState::SUBPOS_MID, m_TTC_mid_r->at(p).at(i));
          // extrapol.set_z(i,TFCSExtrapolationState::SUBPOS_MID, m_TTC_mid_z->at(p).at(i));
          extrapol.set_OK( i, TFCSExtrapolationState::SUBPOS_MID, true );
          extrapol.set_eta(
              i, TFCSExtrapolationState::SUBPOS_MID,
              0.5 * ( ( *m_truthCollection )[0].TTC_entrance_eta[i] + ( *m_truthCollection )[0].TTC_back_eta[i] ) );
          extrapol.set_phi(
              i, TFCSExtrapolationState::SUBPOS_MID,
              0.5 * ( ( *m_truthCollection )[0].TTC_entrance_phi[i] + ( *m_truthCollection )[0].TTC_back_phi[i] ) );
          extrapol.set_r(
              i, TFCSExtrapolationState::SUBPOS_MID,
              0.5 * ( ( *m_truthCollection )[0].TTC_entrance_r[i] + ( *m_truthCollection )[0].TTC_back_r[i] ) );
          extrapol.set_z(
              i, TFCSExtrapolationState::SUBPOS_MID,
              0.5 * ( ( *m_truthCollection )[0].TTC_entrance_z[i] + ( *m_truthCollection )[0].TTC_back_z[i] ) );
        }
      } else {
        if ( m_TTC_IDCaloBoundary_eta->size() > 0 ) {
          extrapol.set_IDCaloBoundary_eta( m_TTC_IDCaloBoundary_eta->at( p ) );
          extrapol.set_IDCaloBoundary_phi( m_TTC_IDCaloBoundary_phi->at( p ) );
          extrapol.set_IDCaloBoundary_r( m_TTC_IDCaloBoundary_r->at( p ) );
          extrapol.set_IDCaloBoundary_z( m_TTC_IDCaloBoundary_z->at( p ) );

          if ( std::isnan( m_TTC_IDCaloBoundary_eta->at( p ) ) || std::abs( m_TTC_IDCaloBoundary_eta->at( p ) ) > 5 ) {
            continue;
          }
        }

        TTC_eta = ( ( *m_TTC_entrance_eta ).at( p ).at( layer ) + ( *m_TTC_back_eta ).at( p ).at( layer ) ) / 2;

        TTC_phi = ( ( *m_TTC_entrance_phi ).at( p ).at( layer ) + ( *m_TTC_back_phi ).at( p ).at( layer ) ) / 2;
        TTC_r   = ( ( *m_TTC_entrance_r ).at( p ).at( layer ) + ( *m_TTC_back_r ).at( p ).at( layer ) ) / 2;
        TTC_z   = ( ( *m_TTC_entrance_z ).at( p ).at( layer ) + ( *m_TTC_back_z ).at( p ).at( layer ) ) / 2;

        for ( int i = 0; i < CaloCell_ID_FCS::MaxSample; ++i ) {
          //          if(m_total_layer_cell_energy[i]==0) continue;
          // extrapol.set_OK(i,TFCSExtrapolationState::SUBPOS_ENT, true);
          extrapol.set_OK( i, TFCSExtrapolationState::SUBPOS_ENT, m_TTC_entrance_OK->at( p ).at( i ) );
          extrapol.set_eta( i, TFCSExtrapolationState::SUBPOS_ENT, m_TTC_entrance_eta->at( p ).at( i ) );
          extrapol.set_phi( i, TFCSExtrapolationState::SUBPOS_ENT, m_TTC_entrance_phi->at( p ).at( i ) );
          extrapol.set_r( i, TFCSExtrapolationState::SUBPOS_ENT, m_TTC_entrance_r->at( p ).at( i ) );
          extrapol.set_z( i, TFCSExtrapolationState::SUBPOS_ENT, m_TTC_entrance_z->at( p ).at( i ) );

          // extrapol.set_OK(i,TFCSExtrapolationState::SUBPOS_EXT, true);
          extrapol.set_OK( i, TFCSExtrapolationState::SUBPOS_EXT, m_TTC_back_OK->at( p ).at( i ) );
          extrapol.set_eta( i, TFCSExtrapolationState::SUBPOS_EXT, m_TTC_back_eta->at( p ).at( i ) );
          extrapol.set_phi( i, TFCSExtrapolationState::SUBPOS_EXT, m_TTC_back_phi->at( p ).at( i ) );
          extrapol.set_r( i, TFCSExtrapolationState::SUBPOS_EXT, m_TTC_back_r->at( p ).at( i ) );
          extrapol.set_z( i, TFCSExtrapolationState::SUBPOS_EXT, m_TTC_back_z->at( p ).at( i ) );

          /*
          //extrapol.set_OK(i,TFCSExtrapolationState::SUBPOS_MID, true);
          extrapol.set_OK(i,TFCSExtrapolationState::SUBPOS_MID, m_TTC_mid_OK->at(p).at(i));
          extrapol.set_eta(i,TFCSExtrapolationState::SUBPOS_MID, m_TTC_mid_eta->at(p).at(i));
          extrapol.set_phi(i,TFCSExtrapolationState::SUBPOS_MID, m_TTC_mid_phi->at(p).at(i));
          extrapol.set_r(i,TFCSExtrapolationState::SUBPOS_MID, m_TTC_mid_r->at(p).at(i));
          extrapol.set_z(i,TFCSExtrapolationState::SUBPOS_MID, m_TTC_mid_z->at(p).at(i));
          */

          extrapol.set_OK( i, TFCSExtrapolationState::SUBPOS_MID,
                           ( extrapol.OK( i, TFCSExtrapolationState::SUBPOS_ENT ) &&
                             extrapol.OK( i, TFCSExtrapolationState::SUBPOS_EXT ) ) );
          extrapol.set_eta( i, TFCSExtrapolationState::SUBPOS_MID,
                            0.5 * ( m_TTC_entrance_eta->at( p ).at( i ) + m_TTC_back_eta->at( p ).at( i ) ) );
          extrapol.set_phi( i, TFCSExtrapolationState::SUBPOS_MID,
                            0.5 * ( m_TTC_entrance_phi->at( p ).at( i ) + m_TTC_back_phi->at( p ).at( i ) ) );
          extrapol.set_r( i, TFCSExtrapolationState::SUBPOS_MID,
                          0.5 * ( m_TTC_entrance_r->at( p ).at( i ) + m_TTC_back_r->at( p ).at( i ) ) );
          extrapol.set_z( i, TFCSExtrapolationState::SUBPOS_MID,
                          0.5 * ( m_TTC_entrance_z->at( p ).at( i ) + m_TTC_back_z->at( p ).at( i ) ) );
        }
      }
      if ( m_debug >= 1 ) extrapol.Print();

      if ( m_debug == 2 )
        std::cout << "TTC eta, phi, r, z = " << TTC_eta << " , " << TTC_phi << " , " << TTC_r << " , " << TTC_z
                  << std::endl;

      if ( pcabin >= 0 )
        if ( pca() != pcabin ) continue;

      ///////////////////////////////////
      //// run simulation chain
      ///////////////////////////////////

      auto t5 = std::chrono::system_clock::now();
      t_bc += t5 - t4;
      int ii = 0;
      for ( auto& validation : m_validations ) {

        auto s = std::chrono::system_clock::now();
        if ( m_debug >= 1 ) {
          std::cout << "Simulate : " << validation.basesim()->GetTitle() << " event=" << ievent
                    << " E=" << total_energy() << " Ebin=" << pca() << std::endl;
          std::cout << "Simulate : " << typeid( *( validation.basesim() ) ).name()
                    << " Title: " << validation.basesim()->GetTitle() << " event=" << ievent << " E=" << total_energy()
                    << " Ebin=" << pca() << " validation: " << typeid( validation ).name()
                    << " Pointer: " << &validation << " Title: " << validation.GetTitle() << std::endl;
        }
        validation.simul().emplace_back( m_randEngine );
        
        auto m = std::chrono::system_clock::now();
        TFCSSimulationState& chain_simul = validation.simul().back();
#ifdef USE_GPU
        chain_simul.set_gpu_rand( m_rd4h );
        chain_simul.set_geold( m_gl );
        chain_simul.set_es( &es );
#endif
        //        std::cout<<"Start simulation of " << typeid(*validation.basesim()).name() <<std::endl ;

        validation.basesim()->simulate( chain_simul, &truthTLV, &extrapol );
        if ( m_debug >= 1 ) {
          chain_simul.Print();
          std::cout << "End simulate : " << validation.basesim()->GetTitle() << " event=" << ievent << std::endl
                    << std::endl;
        }
        auto e = std::chrono::system_clock::now();
        t_c[ii++] += e - s;
        t_c1[ii]  += m - s;
        t_c2[ii]  += e - m;
      }
    } // end loop over particles
  }   // end loop over events
  // auto t_04 = std::chrono::system_clock::now();
#ifdef USE_GPU
  if ( m_rd4h ) CaloGpuGeneral::Rand4Hits_finish( m_rd4h );
#endif

#ifdef USE_OMPGPU
  if ( m_gl->UnloadGpu_omp() ) std::cout << "Successfully unload geometry from GPU!" << std::endl;
#endif

  auto                          t3    = std::chrono::system_clock::now();
  std::chrono::duration<double> diff1 = t3 - t2;
  diff                                = t_01 - start;
  std::cout << "Time of  LoadGeo cpu IO:" << diff.count() << " s" << std::endl;
  diff                                = t2 - t_01;
  std::cout << "Time of InitInputTree, Truth, Extrapol., Valid. :" << diff.count() << " s" << std::endl;
#ifdef USE_GPU
  diff = t1 - t_01;
  std::cout << "Time of  GPU GeoLg() :" << diff.count() << " s" << std::endl;
  diff = t_02 - t1;
  std::cout << "Time of  InitInputTree :" << diff.count() << " s" << std::endl;
  diff = t_03 - t_02;
  std::cout << "Time of  resizeTruth :" << diff.count() << " s" << std::endl;
  std::cout << "Time of  eventloop GPU load FH  :" << time_o1.count() << " s" << std::endl;
#endif
  std::cout << "Time of  eventloop  :" << diff1.count() << " s" << std::endl;
  //std::cout << "Time of  eventloop  GPU ChainA:" << time_g1.count() << " s" << std::endl;
  //std::cout << "Time of  eventloop  GPU ChainB:" << time_g2.count() << " s" << std::endl;
  //std::cout << "Time of  eventloop  host Chain0:" << time_h.count() << " s" << std::endl;
  std::cout << "Time of  eventloop  I/O read from tree:" << t_io.count() << " s" << std::endl;
  std::cout << "Time of  eventloop  before chain simul:" << t_bc.count() << " s" << std::endl;

  for ( int ii = 0; ii < 5; ii++ ) {
    std::cout << "Time for Chain " << ii << " is " << t_c[ii].count()  << " s" << std::endl;
    std::cout << "-- Time for emplace_back is " << t_c1[ii].count() << " s" << std::endl;
    std::cout << "-- Time for simulate is     " << t_c2[ii].count() << " s" << std::endl;
  }
  std::cout << "Time of eventloop LateralShapeParamHitChain :" << time_o2.count() << " s" << std::endl;
  std::cout << "Time of get_number_of_hits :" << time_nhits.count() << " s" << std::endl;
  std::cout << "Time of mchain loop :" << time_mchain.count() << " s" << std::endl;
  std::cout << "Time of hitsim->simulate_hit :" << time_hitsim.count() << " s" << std::endl;
#ifdef FCS_DEBUG
  m_chain->GetTree()->PrintCacheStats();
#endif

  /*
    TCanvas* c;
    c=new TCanvas(hist_cellSFvsE->GetName(),hist_cellSFvsE->GetTitle());
    hist_cellSFvsE->Draw();
    c->SaveAs(".png");

    c=new TCanvas(hist_cellEvsdxdy_org->GetName(),hist_cellEvsdxdy_org->GetTitle());
    hist_cellEvsdxdy_org->SetMaximum(1);
    hist_cellEvsdxdy_org->SetMinimum(0.00001);
    hist_cellEvsdxdy_org->Draw("colz");
    c->SetLogz(true);
    c->SaveAs(".png");

    c=new TCanvas(hist_cellEvsdxdy_sim->GetName(),hist_cellEvsdxdy_sim->GetTitle());
    hist_cellEvsdxdy_sim->SetMaximum(1);
    hist_cellEvsdxdy_sim->SetMinimum(0.00001);
    hist_cellEvsdxdy_sim->Draw("colz");
    c->SetLogz(true);
    c->SaveAs(".png");

    c=new TCanvas(hist_cellEvsdxdy_ratio->GetName(),hist_cellEvsdxdy_ratio->GetTitle());
    hist_cellEvsdxdy_ratio->Draw("colz");
    hist_cellEvsdxdy_ratio->SetMaximum(1.0*8);
    hist_cellEvsdxdy_ratio->SetMinimum(1.0/8);
    c->SetLogz(true);
    c->SaveAs(".png");
  */
}

#if defined USE_GPU || defined USE_OMPGPU
void TFCSShapeValidation::GeoLg() {
  m_gl = new GeoLoadGpu();
  m_gl->set_ncells( m_geo->get_cells()->size() );
  m_gl->set_max_sample( CaloGeometry::MAX_SAMPLING );
  int nrgns = m_geo->get_tot_regions();

  std::cout << "Total GeoRegions= " << nrgns << std::endl;
  std::cout << "Total cells= " << m_geo->get_cells()->size() << std::endl;

  m_gl->set_nregions( nrgns );
  m_gl->set_cellmap( m_geo->get_cells() );

  GeoRegion* GR_ptr = (GeoRegion*)malloc( nrgns * sizeof( GeoRegion ) );
  m_gl->set_regions( GR_ptr );

  Rg_Sample_Index* si = (Rg_Sample_Index*)malloc( CaloGeometry::MAX_SAMPLING * sizeof( Rg_Sample_Index ) );

  m_gl->set_sample_index_h( si );

  int i = 0;
  for ( int is = 0; is < CaloGeometry::MAX_SAMPLING; ++is ) {
    si[is].index = i;
    int nr       = m_geo->get_n_regions( is );
    si[is].size  = nr;
    for ( int ir = 0; ir < nr; ++ir ) region_data_cpy( m_geo->get_region( is, ir ), &GR_ptr[i++] );
    //    std::cout<<"Sample " << is << "regions: "<< nr << ", Region Index " << i << std::endl ;
  }
}

void TFCSShapeValidation::region_data_cpy( CaloGeometryLookup* glkup, GeoRegion* gr ) {

  // Copy all parameters
  gr->set_xy_grid_adjustment_factor( glkup->xy_grid_adjustment_factor() );
  gr->set_index( glkup->index() );

  int neta = glkup->cell_grid_eta();
  int nphi = glkup->cell_grid_phi();
  // std::cout << " copy region " << glkup->index() << "neta= " << neta<< ", nphi= "<<nphi<< std::endl ;

  gr->set_cell_grid_eta( neta );
  gr->set_cell_grid_phi( nphi );

  gr->set_mineta( glkup->mineta() );
  gr->set_minphi( glkup->minphi() );
  gr->set_maxeta( glkup->maxeta() );
  gr->set_maxphi( glkup->maxphi() );

  gr->set_mineta_raw( glkup->mineta_raw() );
  gr->set_minphi_raw( glkup->minphi_raw() );
  gr->set_maxeta_raw( glkup->maxeta_raw() );
  gr->set_maxphi_raw( glkup->maxphi_raw() );

  gr->set_mineta_correction( glkup->mineta_correction() );
  gr->set_minphi_correction( glkup->minphi_correction() );
  gr->set_maxeta_correction( glkup->maxeta_correction() );
  gr->set_maxphi_correction( glkup->maxphi_correction() );

  gr->set_eta_correction( glkup->eta_correction() );
  gr->set_phi_correction( glkup->phi_correction() );
  gr->set_deta( glkup->deta() );
  gr->set_dphi( glkup->dphi() );

  gr->set_deta_double( glkup->deta_double() );
  gr->set_dphi_double( glkup->dphi_double() );

  // now cell array copy from GeoLookup Object
  // new cell_grid is a unsigned long array
  long long* cells = (long long*)malloc( sizeof( long long ) * neta * nphi );
  gr->set_cell_grid( cells );

  if ( neta != (int)( *( glkup->cell_grid() ) ).size() )
    std::cout << "neta " << neta << ", vector eta size " << ( *( glkup->cell_grid() ) ).size() << std::endl;
  for ( int ie = 0; ie < neta; ++ie ) {
    //    	if(nphi != (*(glkup->cell_grid()))[ie].size() )
    //		 std::cout<<"neta " << neta << "nphi "<<nphi <<", vector phi size "<<  (*(glkup->cell_grid()))[ie].size()
    //<< std::endl;

    for ( int ip = 0; ip < nphi; ++ip ) {

      //	if(glkup->index()==0 ) std::cout<<"in loop.."<< ie << " " <<ip << std::endl;
      auto c = ( *( glkup->cell_grid() ) )[ie][ip];
      if ( c ) {
        cells[ie * nphi + ip] = c->calo_hash();

      } else {
        cells[ie * nphi + ip] = -1;
        //	        std::cout<<"NUll cell in loop.."<< ie << " " <<ip << std::endl;
      }
    }
  }
}

#endif

//#ifdef USE_OMPGPU
//bool GeoLoadGpu::LoadGpu_omp() {
//
//  if ( !m_cells || m_ncells == 0 ) {
//    std::cout << "Geometry is empty " << std::endl;
//    return false;
//  }
//
//  int num_devices    = omp_get_num_devices();
//  int initial_device = omp_get_initial_device();
//  int default_device = omp_get_default_device();
//
//  if ( num_devices < 1 or default_device < 0) {
//    std::cout << " ERROR: No device found." << std::endl;
//    return false;
//  }
//
//  GeoGpu geo_gpu_h;
//  
//  // Allocate Device memory for cells and copy cells as array
//  // move cells on host to a array first
//  m_cells_d = (CaloDetDescrElement *) omp_target_alloc( sizeof( CaloDetDescrElement ) * m_ncells, default_device); 
//  if ( m_cells_d == NULL ) {
//    std::cout << " ERROR: No space left on device." << std::endl;
//    return false;
//  }
//
//  std::cout << "omp_target_alloc " << m_ncells << " cells" << std::endl;
//
//  CaloDetDescrElement* cells_Host = (CaloDetDescrElement*)malloc( m_ncells * sizeof( CaloDetDescrElement ) );
//  m_cellid_array                  = (Identifier*)malloc( m_ncells * sizeof( Identifier ) );
//
//  // create an array of cell identities, they are in order of hashids.
//  int ii = 0;
//  for ( t_cellmap::iterator ic = m_cells->begin(); ic != m_cells->end(); ++ic ) {
//     cells_Host[ii]     = *( *ic ).second;
//     Identifier id      = ( ( *ic ).second )->identify();
//     m_cellid_array[ii] = id;  
//     ii++;  
//  }
//  
//  std::size_t offset = 0;
//  ////////////////////////////////
//  // omp_target_memcpy returns zero on success and nonzero on failure.
//  ////////////////////////////////
//  if ( omp_target_memcpy( &m_cells_d[0], cells_Host, sizeof( CaloDetDescrElement ) * m_ncells, 
//                                                offset, offset, default_device, initial_device  ) ) {
//    std::cout << " ERROR: Unable to copy to device." << std::endl;
//    return false;
//  }
//  else {
//    std::cout << "Target device memcpy " << ii << "/" << m_ncells << " cells"
//            << " Total:" << ii * sizeof( CaloDetDescrElement ) << " Bytes" << std::endl;
//  }
//
//  free( cells_Host );
//
////  if ( 0 ) {
////    if ( !SanityCheck() ) { return false; }
////  }
//
//  Rg_Sample_Index* SampleIndex_g;
//  SampleIndex_g = (Rg_Sample_Index *) omp_target_alloc( sizeof( Rg_Sample_Index ) * m_ncells, default_device); 
//  if ( SampleIndex_g == NULL ) {
//    std::cout << " ERROR: No space left on device." << std::endl;;
//    return false;
//  }
//
//  // copy sample_index array  to gpu
//  if ( omp_target_memcpy( SampleIndex_g, m_sample_index_h, sizeof( Rg_Sample_Index ) * m_max_sample, 
//                                                offset, offset, default_device, initial_device ) ) { 
//     std::cout << "ERROR: copy sample index. " << std::endl;
//     return false;
//  }  
//
//  // each Region allocate a grid (long Long) gpu array
//  //  copy array to GPU
//  //  save to regions m_cell_g ;
//  for ( unsigned int ir = 0; ir < m_nregions; ++ir ) {
//    //	std::cout << "debug m_regions_d[ir].cell_grid()[0] " << m_regions[ir].cell_grid()[0] <<std::endl;
//    long long* ptr_g;
//    ptr_g = (long long *) omp_target_alloc( sizeof( long long ) * m_regions[ir].cell_grid_eta() *
//                                                        m_regions[ir].cell_grid_phi(), default_device); 
//    if ( ptr_g == NULL ) {
//      std::cout << " ERROR: No space left on device." << std::endl;;
//      return false;
//    }
//
//    //      std::cout<< "cuMalloc region grid "<<  ir  << std::endl;
//    if ( omp_target_memcpy( ptr_g, m_regions[ir].cell_grid(), sizeof( long long ) * m_regions[ir].cell_grid_eta() 
//                              * m_regions[ir].cell_grid_phi(), offset, offset, default_device, initial_device ) ) { 
//      std::cout << "ERROR: copy m_regions. " << std::endl;
//      return false;
//    }  
//    //      std::cout<< "cpy grid "<<  ir  << std::endl;
//    m_regions[ir].set_cell_grid_g( ptr_g );
//    m_regions[ir].set_all_cells( m_cells_d ); // set this so all region instance know where the GPU cells are, before
//                                              // copy to GPU
//    //	std::cout<<"Gpu cell Pintor in region: " <<m_cells_d << " m_regions[ir].all_cells() " <<
//    // m_regions[ir].all_cells() << std::endl ;
//  }
//
//  // GPU allocate Regions data  and load them to GPU as array of regions
//  m_regions_d = (GeoRegion *) omp_target_alloc( sizeof( GeoRegion ) * m_nregions, default_device); 
//  if ( m_regions_d == NULL ) {
//    std::cout << " ERROR: No space left on device." << std::endl;;
//    return false;
//  }
//  if ( omp_target_memcpy( m_regions_d, m_regions, sizeof( GeoRegion ) * m_nregions,
//                                    offset, offset, default_device, initial_device ) ) { 
//    std::cout << "ERROR: copy m_regions. " << std::endl;
//    return false;
//  } 
////        std::cout<< "Regions Array Copied , size (Byte) " <<  sizeof(GeoRegion)*m_nregions << "sizeof cell *" <<
////        sizeof(CaloDetDescrElement *) << std::endl; std::cout<< "Region Pointer GPU print from host" <<  m_regions_d
////        << std::endl;
//
//  geo_gpu_h.cells        = m_cells_d;
//  geo_gpu_h.ncells       = m_ncells;
//  geo_gpu_h.nregions     = m_nregions;
//  geo_gpu_h.regions      = m_regions_d;
//  geo_gpu_h.max_sample   = m_max_sample;
//  geo_gpu_h.sample_index = SampleIndex_g;
//
//  // Now copy this to GPU and set the static member to this pointer
//  GeoGpu* Gptr;
//  Gptr = (GeoGpu *) omp_target_alloc( sizeof( GeoGpu ), default_device); 
//  if ( Gptr == NULL ) {
//    std::cout << " ERROR: No space left on device." << std::endl;;
//    return false;
//  }
//  if ( omp_target_memcpy( Gptr, &geo_gpu_h, sizeof( GeoGpu ),
//               offset, offset, default_device, initial_device ) ) { 
//    std::cout << "ERROR: copy Gptr. " << std::endl;
//    return false;
//  } 
//
//  //  Geo_g = Gptr;
//  m_geo_d = Gptr;
//
//  // more test for region grids
//  if ( 0 ) { return TestGeo(); }
//  // Free memory allocate on device?
//  return true;
//}
//
//
//#endif
