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
#endif

#ifdef USE_KOKKOS
#  include <Kokkos_Core.hpp>
#endif

std::chrono::duration<double> TFCSShapeValidation::time_o1;
std::chrono::duration<double> TFCSShapeValidation::time_o2;
std::chrono::duration<double> TFCSShapeValidation::time_g1;
std::chrono::duration<double> TFCSShapeValidation::time_g2;
std::chrono::duration<double> TFCSShapeValidation::time_h;

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
  m_randEngine->setSeed( seed );
#if defined USE_GPU || defined USE_OMPGPU
  auto t0                            = std::chrono::system_clock::now();
  m_gl                               = 0;
  m_rd4h                             = CaloGpuGeneral::Rand4Hits_init( MAXHITS, MAXBINS, seed, true );
  auto                          t1   = std::chrono::system_clock::now();
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

  std::chrono::duration<double> t_c[5] = {std::chrono::duration<double, std::ratio<1>>::zero()};
  std::chrono::duration<double> t_bc   = std::chrono::duration<double, std::ratio<1>>::zero();
  std::chrono::duration<double> t_io   = std::chrono::duration<double, std::ratio<1>>::zero();

#if defined USE_GPU || defined USE_OMPGPU

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
#endif

  int nentries = m_nentries;
  int layer    = m_layer;
  std::cout << "TFCSShapeValidation::LoopEvents(): Running on layer = " << layer << ", pcabin = " << pcabin
            << std::endl;

  InitInputTree( m_chain, layer );
  auto t_02 = std::chrono::system_clock::now();

  ///////////////////////////////////
  //// Initialize truth, extraplolation and all validation structures
  ///////////////////////////////////
  m_truthTLV.resize( nentries );
  m_extrapol.resize( nentries );
  auto t_03 = std::chrono::system_clock::now();

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

#if defined USE_GPU || defined USE_OMPGPU
  TFCSSimulationState::EventStatus es    = {-1, 0, 0, 0, 0, 0, 0, false, nullptr, nullptr, 0, false, false};
  long int                         index = 0, tot_hits = 0;
  long int                         bin_index = 0;
  int                              n_simbins = 0;
  int                              iv        = 0;
  int                              g_sims_v[MAX_SIM];
  int                              g_sims_st[MAX_SIM];
  long                             simbins[MAX_SIMBINS]; // max simulbins
  HitParams hitparams[MAX_SIMBINS]; // this array is holding hit paramters for each bin(event/particle/validation/cs)
  es.hitparams                            = (void*)&( hitparams[0] );
  es.simbins                              = &( simbins[0] );
  es.n_simbins                            = 0;
  std::chrono::duration<double> t_g_sim   = std::chrono::duration<double, std::ratio<1>>::zero();
  std::chrono::duration<double> t_g_sim_A = std::chrono::duration<double, std::ratio<1>>::zero();
  std::chrono::duration<double> t_g_sim_B = std::chrono::duration<double, std::ratio<1>>::zero();
#endif
  std::chrono::duration<double> t_st = std::chrono::duration<double, std::ratio<1>>::zero();
  auto                          t2   = std::chrono::system_clock::now();
  //for ( int ievent = m_firstevent; ievent < nentries; ievent++ ) {
  for ( int ievent = m_firstevent; ievent < 250; ievent++ ) {
    printf("OpenMP1 Event %d \n", ievent);

#if defined USE_GPU || defined USE_OMPGPU

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
#if defined USE_GPU || defined USE_OMPGPU
      iv = 0;
#endif
      int ii = 0;
      for ( auto& validation : m_validations ) {

        auto ss = std::chrono::system_clock::now();
        if ( m_debug >= 1 ) {
          std::cout << "Simulate : " << validation.basesim()->GetTitle() << " event=" << ievent
                    << " E=" << total_energy() << " Ebin=" << pca() << std::endl;
          std::cout << "Simulate : "
                    << "validation:" << typeid( validation ).name() << ", "
                    << typeid( *( validation.basesim() ) ).name() << " Title: " << validation.basesim()->GetTitle()
                    << " event=" << ievent << " E=" << total_energy() << " Ebin=" << pca()
                    << " validation: " << typeid( validation ).name() << " Pointer: " << &validation
                    << " Title: " << validation.GetTitle() << std::endl;
        }
#if defined USE_GPU || defined USE_OMPGPU
        int vi = iv;
        int si = validation.simul().size();
#endif
        auto te1 = std::chrono::system_clock::now();
        validation.simul().emplace_back( m_randEngine );
        TFCSSimulationState& chain_simul = validation.simul().back();
        auto                 te2         = std::chrono::system_clock::now();
        t_st += te2 - te1;
#if defined USE_GPU || defined USE_OMPGPU
        es.ip        = p;
        es.ievent    = ievent;
        es.iv        = iv;
        es.tot_hits  = tot_hits;
        es.index     = index;
        es.bin_index = bin_index;
        es.n_simbins = n_simbins;
        es.gpu       = false;
        if ( p != 0 ) es.is_first = false;
        if ( p == ( particles - 1 ) && ievent == ( nentries - 1 ) )
          es.is_last = true;
        else
          es.is_last = false;

        chain_simul.set_gpu_rand( m_rd4h );
        chain_simul.set_geold( m_gl );
        chain_simul.set_es( es );
#endif
        validation.basesim()->simulate( chain_simul, &truthTLV, &extrapol );
        if ( m_debug >= 1 ) {
          chain_simul.Print();
          std::cout << "End simulate : " << validation.basesim()->GetTitle() << " event=" << ievent << std::endl
                    << std::endl;
        }
#if defined USE_GPU || defined USE_OMPGPU
        iv++;
        if ( ( *chain_simul.get_es() ).gpu == true ) {
          g_sims_v[index]  = vi;
          g_sims_st[index] = si;
          index++; // index is particle/event which use GPU is some sample
        }
        //       else std::cout<<"Skipping GPU for E/P: " << ievent<< " , "<< p << std::endl ;

        bin_index = ( *( chain_simul.get_es() ) ).bin_index;
        tot_hits  = ( *( chain_simul.get_es() ) ).tot_hits;
        n_simbins = ( *( chain_simul.get_es() ) ).n_simbins;
        if ( index >= MAX_SIM || tot_hits > ( MAXHITS - 100000 ) || es.is_last ) {

          //   here need to do GPU simulation !!!!.
          
          auto tg_s = std::chrono::system_clock::now();
          CaloGpuGeneral::load_hitsim_params( m_rd4h, &( hitparams[0] ), &( simbins[0] ), n_simbins );
          auto tg_s_A = std::chrono::system_clock::now();
          t_g_sim_A += tg_s_A - tg_s;

          Sim_Args args;
          args.debug        = m_debug;
          args.rd4h         = m_rd4h;
          args.geo          = GeoLoadGpu::Geo_g;
          args.cells_energy = nullptr;
          args.hitcells_E   = nullptr;
          args.hitcells_E_h = nullptr;
          args.ct           = nullptr;
          args.ct_h         = nullptr;
          args.hitparams    = nullptr;
          args.hitparams_h  = hitparams;
          args.simbins      = nullptr;
          args.nbins        = n_simbins;
          args.nsims        = index;
          args.nhits        = tot_hits;
          args.ncells       = GeoLoadGpu::num_cells;


          

          CaloGpuGeneral::simulate_hits_gr( args );
          auto tg_s_B = std::chrono::system_clock::now();
          t_g_sim_B += tg_s_B - tg_s;

          
          for ( int isim = 0; isim < index; isim++ ) {
            TFCSSimulationState& sim = m_validations[g_sims_v[isim]].simul()[g_sims_st[isim]];
            //  std::cout << "gpucellCT["<<isim<<"]=" << args.ct_h[isim] <<std::endl ;
            for ( int ii = 0; ii < args.ct_h[isim]; ii++ ) {
              // if(args.hitcells_E_h[ii+isim*MAXHITCT].cellid >200000 || args.hitcells_E_h[ii+isim*MAXHITCT].cellid <=0
              // ) std::cout << "Something Wrong cellid: " << args.hitcells_E_h[ii+isim*MAXHITCT].cellid <<",
              // isim="<<isim <<", ii="<<ii << std::endl ; std::cout << "Id:" <<
              // args.hitcells_E_h[ii+isim*MAXHITCT].cellid ;
              const CaloDetDescrElement* cellele = m_gl->index2cell( args.hitcells_E_h[ii + isim * MAXHITCT].cellid );
              // std::cout << ",Is" << isim <<"Ic"<<ii;
              sim.deposit( cellele, args.hitcells_E_h[ii + isim * MAXHITCT].energy );
            }
          }
          //   std::cout<<std::endl;

          auto tg_e = std::chrono::system_clock::now();
          t_g_sim += tg_e - tg_s;

          //		std::cout<<"reset after Event: "<< ievent<<" partical "<< p << ",total Hits: " << tot_hits
          //		<<", Index="<< index <<" , N_simbins="<< n_simbins<<std::endl;
          index     = 0;
          tot_hits  = 0;
          n_simbins = 0;
        }

#endif
        auto e = std::chrono::system_clock::now();
        t_c[ii++] += e - ss;

      } // validation loop

    } // end loop over particles
  }   // end loop over events
  // auto t_04 = std::chrono::system_clock::now();
#if defined USE_GPU || defined USE_OMPGPU
  if ( m_rd4h ) CaloGpuGeneral::Rand4Hits_finish( m_rd4h );
#endif

  auto                          t3    = std::chrono::system_clock::now();
  std::chrono::duration<double> diff1 = t3 - t2;
  diff                                = t_01 - start;
  std::cout << "Time of  LoadGeo cpu IO:" << diff.count() << " s" << std::endl;
#if defined USE_GPU || defined USE_OMPGPU
  diff = t1 - t_01;
  std::cout << "Time of  GeoLg() :" << diff.count() << " s" << std::endl;
  diff = t_02 - t1;
  std::cout << "Time of  InitInputTree :" << diff.count() << " s" << std::endl;
  diff = t_03 - t_02;
  std::cout << "Time of  resizeTruth :" << diff.count() << " s" << std::endl;
  std::cout << "Time of  eventloop GPU load FH  :" << time_o1.count() << " s" << std::endl;
#endif
  std::cout << "Time of  eventloop  simulation state creation  :" << t_st.count() << " s" << std::endl;
  std::cout << "Time of  eventloop  LateralShapeParamHitChain  :" << time_o2.count() << " s" << std::endl;
  std::cout << "Time of  eventloop  :" << diff1.count() << " s" << std::endl;
  std::cout << "Time of  eventloop  GPU ChainA:" << time_g1.count() << " s" << std::endl;
  std::cout << "Time of  eventloop  GPU ChainB:" << time_g2.count() << " s" << std::endl;
  std::cout << "Time of  eventloop  host Chain0:" << time_h.count() << " s" << std::endl;
  std::cout << "Time of  eventloop  before chain simul:" << t_bc.count() << " s" << std::endl;
  std::cout << "Time of  eventloop  I/O read from tree:" << t_io.count() << " s" << std::endl;
#if defined USE_GPU || defined USE_OMPGPU
  std::cout << "Time of  GPU group Sim_total:" << t_g_sim.count() << " s" << std::endl;
  std::cout << "Time of  GPU copy params:" << t_g_sim_A.count() << " s" << std::endl;
  std::cout << "Time of  GPU Sim:" << t_g_sim_B.count() << " s" << std::endl;
#endif
  for ( int ii = 0; ii < 5; ii++ )
    std::cout << "Time for Chain " << ii << " is " << t_c[ii].count() << " s" << std::endl;

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

  GeoRegion* GR_ptr = m_gl->get_regions();
  if(!GR_ptr) {
    GR_ptr = (GeoRegion*)malloc( nrgns * sizeof( GeoRegion ) );
  }
  m_gl->set_regions( GR_ptr );

  Rg_Sample_Index* si = m_gl->get_sample_index_h();
  if(!si) {
    si = (Rg_Sample_Index*)malloc( CaloGeometry::MAX_SAMPLING * sizeof( Rg_Sample_Index ) );
  }

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

  size_t neta = glkup->cell_grid_eta();
  int    nphi = glkup->cell_grid_phi();
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
  long long* cells = m_gl->get_cell_grid(neta,nphi);
  if(!cells) {
    cells = (long long*)malloc( sizeof( long long ) * neta * nphi );
  }
  gr->set_cell_grid( cells );

  if ( neta != ( *( glkup->cell_grid() ) ).size() )
    std::cout << "neta " << neta << ", vector eta size " << ( *( glkup->cell_grid() ) ).size() << std::endl;
  for ( size_t ie = 0; ie < neta; ++ie ) {
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
