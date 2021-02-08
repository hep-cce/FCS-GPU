/*
  Copyright (C) 2002-2017 CERN for the benefit of the ATLAS collaboration
 */

#include "FastCaloSimAnalyzer/TFCSFlatNtupleMaker.h"

#include "atlasrootstyle/AtlasLabels.h"
#include "atlasrootstyle/AtlasStyle.h"
#include "atlasrootstyle/AtlasUtils.h"

#include "TStyle.h"
#include "TString.h"
#include "TMath.h"
#include "TFile.h"
#include "TH1.h"
#include "TH2.h"
#include "TROOT.h"
#include "TCanvas.h"
#include "TChain.h"

#include <iostream>
#include <tuple>
#include <algorithm>
#include <fstream>

TFCSFlatNtupleMaker::TFCSFlatNtupleMaker() { m_debug = 0; }

TFCSFlatNtupleMaker::TFCSFlatNtupleMaker( TChain* chain, TString outputfilename, std::vector<int> vlayer ) {

  m_debug  = 0;
  m_chain  = chain;
  m_output = outputfilename;
  m_vlayer = vlayer;
}

TFCSFlatNtupleMaker::~TFCSFlatNtupleMaker() {}

void TFCSFlatNtupleMaker::LoopEvents() {

  int              nentries    = m_nentries;
  std::vector<int> v_relLayers = m_vlayer;

  auto f_out = std::unique_ptr<TFile>( TFile::Open( m_output.c_str(), "recreate" ) );
  if ( !f_out ) {
    std::cerr << "Error: Could not create file '" << m_output << "'" << std::endl;
    return;
  }

  f_out->SetCompressionAlgorithm( 2 );
  // f_out->SetCompressionSettings(ROOT::CompressionSettings(ROOT::kLZMA, 9));
  f_out->SetCompressionLevel( 9 );
  TTree* t_out = new TTree( "FCS_flatNtuple", "mini_calohit" );
  BookFlatNtuple( t_out );

  for ( unsigned int ilayer = 0; ilayer < v_relLayers.size(); ilayer++ ) {

    int layer = v_relLayers.at( ilayer );
    // bool isBarrel = (layer >= 0 && layer <= 3) || (layer >= 12 && layer <= 20);

    std::cout << " * Running on layer = " << layer << std::endl;

    InitInputTree( m_chain, layer );

    for ( int ievent = 0; ievent < nentries; ievent++ ) {
      if ( ievent % 1000 == 0 ) std::cout << " Event: " << ievent << std::endl;

      m_chain->GetEntry( ievent );

      float px = m_truthPx->at( 0 );
      float py = m_truthPy->at( 0 );
      float pz = m_truthPz->at( 0 );
      float E  = m_truthE->at( 0 );

      TLorentzVector truthTLV;
      truthTLV.SetPxPyPzE( px, py, pz, E );

      float truth_eta = truthTLV.Eta();
      float truth_phi = truthTLV.Phi();

      int pca = m_pca;

      if ( m_debug == 2 && ievent < 100 ) {
        std::cout << " pca = " << pca << std::endl;
        std::cout << " truth eta = " << truth_eta << std::endl;
        std::cout << " truth phi = " << truth_phi << std::endl;
      }

      float TTC_eta, TTC_phi, TTC_r, TTC_z;

      if ( !m_isNewSample ) {
        TTC_eta = ( *m_truthCollection )[0].TTC_entrance_eta[0];
        TTC_phi = ( *m_truthCollection )[0].TTC_entrance_phi[0];
        TTC_r   = ( *m_truthCollection )[0].TTC_entrance_r[0];
        TTC_z   = ( *m_truthCollection )[0].TTC_entrance_z[0];

      } else {

        TTC_eta = ( ( *m_TTC_entrance_eta ).at( 0 ).at( layer ) + ( *m_TTC_back_eta ).at( 0 ).at( layer ) ) / 2;

        TTC_phi = ( ( *m_TTC_entrance_phi ).at( 0 ).at( layer ) + ( *m_TTC_back_phi ).at( 0 ).at( layer ) ) / 2;
        TTC_r   = ( ( *m_TTC_entrance_r ).at( 0 ).at( layer ) + ( *m_TTC_back_r ).at( 0 ).at( layer ) ) / 2;
        TTC_z   = ( ( *m_TTC_entrance_z ).at( 0 ).at( layer ) + ( *m_TTC_back_z ).at( 0 ).at( layer ) ) / 2;
      }

      if ( m_debug == 2 && ievent < 100 )
        std::cout << " TTC eta, phi, z = " << TTC_eta << " , " << TTC_phi << TTC_r << TTC_z << std::endl;

      unsigned int ncells = m_cellVector->size();

      for ( unsigned int icell = 0; icell < ncells; icell++ ) {
        unsigned int nhits = m_cellVector->m_vector.at( icell ).hit.size();

        float cell_energy = m_cellVector->m_vector.at( icell ).cell.energy;

        float scalefactor = m_cellVector->m_vector.at( icell ).scalingfactor();

        for ( unsigned int ihit = 0; ihit < nhits; ihit++ ) {

          float hit_x    = m_cellVector->m_vector.at( icell ).hit.at( ihit ).hit_x;
          float hit_y    = m_cellVector->m_vector.at( icell ).hit.at( ihit ).hit_y;
          float hit_z    = m_cellVector->m_vector.at( icell ).hit.at( ihit ).hit_z;
          float hit_time = m_cellVector->m_vector.at( icell ).hit.at( ihit ).hit_time;

          Long64_t cell_identifier = m_cellVector->m_vector.at( icell ).hit.at( ihit ).cell_identifier;
          Long64_t identifier      = m_cellVector->m_vector.at( icell ).hit.at( ihit ).identifier;
          int      sampling        = m_cellVector->m_vector.at( icell ).hit.at( ihit ).sampling;

          TVector3 hitVector;

          hitVector.SetXYZ( hit_x, hit_y, hit_z );

          float hit_eta = hitVector.Eta();
          float hit_phi = hitVector.Phi();

          float energy = m_cellVector->m_vector.at( icell ).hit.at( ihit ).hit_energy;

          float eta = hit_eta - TTC_eta;
          float phi = DeltaPhi( hit_phi, TTC_phi );

          if ( m_debug == 2 && icell < 10 && ihit < 10 )
            std::cout << "eta, phi, energy = " << eta << ", " << phi << ", " << energy << std::endl;

          float r     = TMath::Sqrt( eta * eta + phi * phi );
          float alpha = TMath::ATan2( phi, eta );

          if ( alpha < 0 ) alpha = 2.0 * TMath::Pi() + alpha;

          float eta_mm, phi_mm;

          // std::tie(eta_mm, phi_mm) = GetUnitsmm(hit_eta, eta, phi, TTC_r, TTC_z);
          std::tie( eta_mm, phi_mm ) = GetUnitsmm( TTC_eta, eta, phi, TTC_r, TTC_z );

          if ( m_debug == 3 && icell < 10 && ihit < 10 ) {
            float eta_mm_v2, phi_mm_v2;
            std::tie( eta_mm_v2, phi_mm_v2 ) = GetUnitsmm( hit_eta, eta, phi, TTC_r, TTC_z );
            std::cout << "sampling: " << sampling << std::endl;
            std::cout << "eta, phi " << eta << ", " << phi << std::endl;
            std::cout << "eta_mm, phi_mm " << eta_mm << ", " << phi_mm << std::endl;
            std::cout << "eta_mm_v2, phi_mm_v2 " << eta_mm_v2 << ", " << phi_mm_v2 << std::endl;
            std::cout << "eta_mm_v2 - eta_mm, phi_mm_v2 - phi_mm " << eta_mm_v2 - eta_mm << ", " << phi_mm_v2 - phi_mm
                      << std::endl;
            std::cout << "(eta_mm_v2 - eta_mm)/eta_mm, (phi_mm_v2 - phi_mm)/phi_mm " << ( eta_mm_v2 - eta_mm ) / eta_mm
                      << ", " << ( phi_mm_v2 - phi_mm ) / phi_mm << std::endl;
          }

          float r_mm     = TMath::Sqrt( eta_mm * eta_mm + phi_mm * phi_mm );
          float alpha_mm = TMath::ATan2( phi_mm, eta_mm );

          if ( alpha_mm < 0 ) alpha_mm = 2.0 * TMath::Pi() + alpha_mm;

          if ( m_debug == 2 && icell < 10 && ihit < 10 ) {

            std::cout << " r, alpha = " << r << ", " << alpha << std::endl;
            std::cout << " r_mm, alpha_mm = " << r_mm << ", " << alpha_mm << std::endl;
            std::cout << " d_phi/d_eta = " << phi / eta << " d_phi_mm/d_eta_mm = " << phi_mm / eta_mm << std::endl;
            std::cout << " m_isNewSample = " << m_isNewSample << std::endl;
          }

          // TBranches
          b_m_ievent = ievent;
          if ( icell == 0 and ihit == 0 )
            b_m_new_event = true;
          else
            b_m_new_event = false;
          if ( ihit == 0 )
            b_m_new_cell = true;
          else
            b_m_new_cell = false;
          b_m_cell_identifier = cell_identifier;
          b_m_identifier      = identifier;
          b_m_layer           = layer;
          b_m_pca             = pca;
          b_m_truth_eta       = truth_eta;
          b_m_truth_phi       = truth_phi;
          b_m_truth_energy    = E;
          b_m_TTC_eta         = TTC_eta;
          b_m_TTC_phi         = TTC_phi;
          b_m_TTC_r           = TTC_r;
          b_m_TTC_z           = TTC_z;
          b_m_deta            = eta;
          b_m_dphi            = phi;
          b_m_deta_mm         = eta_mm;
          b_m_dphi_mm         = phi_mm;
          b_m_energy          = energy;
          b_m_scalefactor     = scalefactor;
          b_m_hit_time        = hit_time;
          b_m_cell_energy     = cell_energy;
          b_m_r               = r;
          b_m_alpha           = alpha;
          b_m_r_mm            = r_mm;
          b_m_alpha_mm        = alpha_mm;
          // b_m_dx= hit_x - TTC_x;
          // b_m_dy= hit_y - TTC_y;
          f_out->cd();

          t_out->Fill();

        } // end loop over hits
      }   // end loop over cells
    }     // end loop over events

    if ( m_cellVector ) delete m_cellVector;
    if ( m_truthCollection ) delete m_truthCollection;
    if ( m_TTC_entrance_eta ) delete m_TTC_entrance_eta;
    if ( m_TTC_entrance_phi ) delete m_TTC_entrance_phi;
    if ( m_TTC_entrance_r ) delete m_TTC_entrance_r;
    if ( m_TTC_entrance_z ) delete m_TTC_entrance_z;
    if ( m_TTC_back_eta ) delete m_TTC_back_eta;
    if ( m_TTC_back_phi ) delete m_TTC_back_phi;
    if ( m_TTC_back_r ) delete m_TTC_back_r;
    if ( m_TTC_back_z ) delete m_TTC_back_z;
  } // end loop over layers

  // t_out->Write();
  f_out->Close();

  std::cout << "flatNtuple at " << m_output.c_str() << std::endl;
}

void TFCSFlatNtupleMaker::StudyHitMerging() {

  gROOT->SetBatch( kTRUE );

#ifdef __CINT__
  gROOT->LoadMacro( "atlasstyle/AtlasLabels.C" );
  gROOT->LoadMacro( "atlasstyle/AtlasUtils.C" );
#endif

  SetAtlasStyle();
  gStyle->SetOptStat( 1 );

  TString label1 = TFCSAnalyzerBase::GetLabel();

  int              nentries    = m_nentries;
  std::vector<int> v_relLayers = m_vlayer;
  std::string      merge       = m_merge;
  std::string      particle    = m_label;

  std::string dir = "plots_hit_" + particle + "_" + merge;
  system( ( "mkdir -p " + dir ).c_str() );

  std::string file = "hit_" + particle + merge + ".root";
  auto        fout = std::unique_ptr<TFile>( TFile::Open( file.c_str(), "recreate" ) );
  if ( !fout ) {
    std::cerr << "Error: Could not create file '" << file << "'" << std::endl;
    return;
  }

  for ( unsigned int ilayer = 0; ilayer < v_relLayers.size(); ilayer++ ) {

    int layer = v_relLayers.at( ilayer );

    std::cout << " * Running on layer = " << layer << std::endl;

    TH1D* hmin  = new TH1D( Form( "hit_diff_min_layer%i", layer ), Form( "hit_diff_min_layer%i", layer ), 500, 0, 5 );
    TH2D* h2min = new TH2D( Form( "hit_time_diff_min_layer%i", layer ), Form( "hit_time_diff_min_layer%i", layer ),
                            3000, 0, 3000, 1000, 0, 10 );

    hmin->Sumw2();
    hmin->SetTitle( "" );
    hmin->GetXaxis()->SetTitle( "min. #Deltahit_{ij}" );

    h2min->GetXaxis()->SetTitle( "min. #Deltatime_{ij}" );
    h2min->GetYaxis()->SetTitle( "min. #Deltahit_{ij}" );

    TCanvas* canvas = new TCanvas( Form( "c_layer%i", layer ), Form( "c_layer%i", layer ), 800, 600 );

    InitInputTree( m_chain, layer );

    for ( int ievent = 0; ievent < nentries; ievent++ ) {
      if ( ievent % 1000 == 0 ) std::cout << " Event: " << ievent << std::endl;

      m_chain->GetEntry( ievent );

      unsigned int ncells = m_cellVector->size();

      for ( unsigned int icell = 0; icell < ncells; icell++ ) {
        unsigned int nhits = m_cellVector->m_vector.at( icell ).hit.size();

        float min_hit  = 1e9;
        float min_time = 1e9;

        if ( nhits > 1 ) {
          for ( unsigned int ihit = 0; ihit < nhits; ihit++ ) {

            float hit_x  = m_cellVector->m_vector.at( icell ).hit.at( ihit ).hit_x;
            float hit_y  = m_cellVector->m_vector.at( icell ).hit.at( ihit ).hit_y;
            float hit_z  = m_cellVector->m_vector.at( icell ).hit.at( ihit ).hit_z;
            float energy = m_cellVector->m_vector.at( icell ).hit.at( ihit ).hit_energy;
            energy       = 1.;

            float hit_time_i = m_cellVector->m_vector.at( icell ).hit.at( ihit ).hit_time * energy;

            Long64_t identifier_i = m_cellVector->m_vector.at( icell ).hit.at( ihit ).identifier;

            // std::cout << " hit time = " << hit_time_i << std::endl ;

            TVector3 vhit_i;
            vhit_i.SetXYZ( hit_x * energy, hit_y * energy, hit_z * energy );

            for ( unsigned int jhit = nhits - 1; jhit > ihit; jhit-- ) {
              Long64_t identifier_j = m_cellVector->m_vector.at( icell ).hit.at( jhit ).identifier;

              if ( identifier_i != identifier_j ) {
                // std::cout << "id_i, id_j = " << identifier_i << " , " << identifier_j << std::endl ;
                continue;
              }

              float hit_x  = m_cellVector->m_vector.at( icell ).hit.at( jhit ).hit_x;
              float hit_y  = m_cellVector->m_vector.at( icell ).hit.at( jhit ).hit_y;
              float hit_z  = m_cellVector->m_vector.at( icell ).hit.at( jhit ).hit_z;
              float energy = m_cellVector->m_vector.at( icell ).hit.at( jhit ).hit_energy;

              energy = 1.;

              float hit_time_j = m_cellVector->m_vector.at( icell ).hit.at( jhit ).hit_time * energy;

              TVector3 vhit_j;
              vhit_j.SetXYZ( hit_x * energy, hit_y * energy, hit_z * energy );

              float diff2 = ( vhit_i.X() - vhit_j.X() ) * ( vhit_i.X() - vhit_j.X() ) +
                            ( vhit_i.Y() - vhit_j.Y() ) * ( vhit_i.Y() - vhit_j.Y() ) +
                            ( vhit_i.Z() - vhit_j.Z() ) * ( vhit_i.Z() - vhit_j.Z() );

              float diff = TMath::Sqrt( diff2 );

              float time_diff = hit_time_i - hit_time_j;

              if ( diff < min_hit ) {
                min_hit  = diff;
                min_time = abs( time_diff );
              }
            } // end loop over jhits
          }   // end loop over ihits
        }
        hmin->Fill( min_hit );
        h2min->Fill( min_time, min_hit );

      } // end loop over cells
    }   // end loop over events

    TString label = Form( "%s, layer %i", label1.Data(), layer );
    canvas->cd();
    h2min->Draw();
    myText( 0.25, 0.96, 1, label );
    ATLASLabel( 0.18, 0.05, "Simulation Internal" );

    canvas->SetLogx();
    canvas->SaveAs( ( dir + "/min_hit_merge_layer" + std::to_string( layer ) + ".png" ).c_str() );
    fout->cd();
    hmin->Write();
    h2min->Write();
    if ( hmin ) delete hmin;
    if ( h2min ) delete h2min;
    if ( canvas ) delete canvas;
  } // end loop over layers
}

void TFCSFlatNtupleMaker::BookFlatNtuple( TTree* t ) {
  b_m_ievent          = -1;
  b_m_new_event       = false;
  b_m_new_cell        = false;
  b_m_cell_identifier = 0;
  b_m_identifier      = 0;
  b_m_layer           = -1;
  b_m_pca             = -1;
  b_m_truth_eta       = 0.;
  b_m_truth_phi       = 0.;
  b_m_truth_energy    = 0.;
  b_m_TTC_eta         = 0.;
  b_m_TTC_phi         = 0.;
  b_m_TTC_r           = 0.;
  b_m_TTC_z           = 0.;
  b_m_deta            = 0.;
  b_m_dphi            = 0.;
  b_m_deta_mm         = 0.;
  b_m_dphi_mm         = 0.;
  b_m_energy          = 0.;
  b_m_scalefactor     = 0.;
  b_m_hit_time        = 0.;
  b_m_cell_energy     = 0.;
  b_m_r               = 0.;
  b_m_alpha           = 0.;
  b_m_r_mm            = 0.;
  b_m_alpha_mm        = 0.;
  // b_m_dx = 0.;
  // b_m_dy = 0.;

  t->Branch( "event_number", &b_m_ievent );
  t->Branch( "is_new_event", &b_m_new_event );
  t->Branch( "is_new_cell", &b_m_new_cell );
  t->Branch( "cell_identifier", &b_m_cell_identifier );
  t->Branch( "identifier", &b_m_identifier );
  t->Branch( "layer", &b_m_layer );
  t->Branch( "pca", &b_m_pca );
  t->Branch( "truth_eta", &b_m_truth_eta );
  t->Branch( "truth_phi", &b_m_truth_phi );
  t->Branch( "truth_energy", &b_m_truth_energy );
  t->Branch( "TTC_eta", &b_m_TTC_eta );
  t->Branch( "TTC_phi", &b_m_TTC_phi );
  t->Branch( "TTC_r", &b_m_TTC_r );
  t->Branch( "TTC_z", &b_m_TTC_z );
  t->Branch( "d_eta", &b_m_deta );
  t->Branch( "d_phi", &b_m_dphi );
  t->Branch( "d_eta_mm", &b_m_deta_mm );
  t->Branch( "d_phi_mm", &b_m_dphi_mm );
  t->Branch( "hit_energy", &b_m_energy );
  t->Branch( "scale_factor", &b_m_scalefactor );
  t->Branch( "hit_time", &b_m_hit_time );
  t->Branch( "cell_energy", &b_m_cell_energy );
  t->Branch( "radius", &b_m_r );
  t->Branch( "alpha", &b_m_alpha );
  t->Branch( "radius_mm", &b_m_r_mm );
  t->Branch( "alpha_mm", &b_m_alpha_mm );
  // t->Branch("d_x", &b_m_dx);
  // t->Branch("d_y", &b_m_dy);
}
