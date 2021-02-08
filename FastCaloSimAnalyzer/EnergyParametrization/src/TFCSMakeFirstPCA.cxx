using namespace std;

#include "TMatrixD.h"
#include "TVectorD.h"
#include "TH1D.h"
#include "TFile.h"
#include "TROOT.h"
#include "TSystem.h"
#include "TApplication.h"
#include "TTree.h"
#include "TSystem.h"
#include "TH2D.h"
#include "TPrincipal.h"
#include "TMath.h"
#include "TBrowser.h"
#include "TFCSMakeFirstPCA.h"
#include "TreeReader.h"
#include "TLorentzVector.h"
#include "TChain.h"
//#include "TRandom3.h"
#include <CLHEP/Random/RandFlat.h>

#include <iostream>

#define LAYERMAX 24

TFCSMakeFirstPCA::TFCSMakeFirstPCA() {
  // default parameters:
  m_use_absolute_layercut = 0;
  m_numberfinebins        = 5000;
  m_edepositcut           = 0.001;
  m_cut_eta_low           = -100;
  m_cut_eta_high          = 100;
  m_apply_etacut          = 1;

  m_dorescale   = 1;
  m_outfilename = "";
  m_chain       = 0;
}

TFCSMakeFirstPCA::TFCSMakeFirstPCA( TChain* chain, string outfilename ) {
  // default parameters:
  m_use_absolute_layercut = 0;
  m_numberfinebins        = 5000;
  m_edepositcut           = 0.001;
  m_cut_eta_low           = -100;
  m_cut_eta_high          = 100;
  m_dorescale             = 1;
  m_outfilename           = outfilename;
  m_chain                 = chain;
}

void TFCSMakeFirstPCA::apply_etacut( int flag ) { m_apply_etacut = flag; }

void TFCSMakeFirstPCA::set_cumulativehistobins( int bins ) { m_numberfinebins = bins; }

void TFCSMakeFirstPCA::set_edepositcut( double cut ) { m_edepositcut = cut; }

void TFCSMakeFirstPCA::set_etacut( double cut_low, double cut_high ) {
  m_cut_eta_low  = cut_low;
  m_cut_eta_high = cut_high;
}

// void TFCSMakeFirstPCA::run()
void TFCSMakeFirstPCA::run( CLHEP::HepRandomEngine* randEngine ) {
  cout << endl;
  cout << "****************" << endl;
  cout << "     1st PCA" << endl;
  cout << "****************" << endl;
  cout << endl;
  cout << "Now running firstPCA with the following parameters:" << endl;
  cout << "  Energy deposit cut: " << m_edepositcut << endl;
  cout << "  Number of bins in the cumulative histograms " << m_numberfinebins << endl;
  cout << "  Eta cut: " << m_cut_eta_low << " " << m_cut_eta_high << endl;
  cout << "  Apply eta cut: " << m_apply_etacut << endl;
  cout << endl;
  TreeReader* read_inputTree = new TreeReader();
  read_inputTree->SetTree( m_chain );

  vector<int>   layerNr     = get_relevantlayers( read_inputTree, m_edepositcut );
  vector<TH1D*> histos_data = get_G4_histos_from_tree( layerNr, read_inputTree );

  TH1I*          h_layer_input = new TH1I( "h_layer_input", "h_layer_input", 24, -0.5, 23.5 );
  vector<string> layer;
  for ( unsigned int l = 0; l < layerNr.size(); l++ ) {
    layer.push_back( Form( "layer%i", layerNr[l] ) );
    h_layer_input->SetBinContent( layerNr[l] + 1, 1 );
  }
  layer.push_back( "totalE" );

  vector<TH1D*> cumul_data = get_cumul_histos( layer, histos_data );

  cout << "--- Now define the TPrincipal" << endl;
  TPrincipal* principal = new TPrincipal( layer.size(), "ND" ); // ND means normalize cov matrix and store data

  TTree* T_Gauss0 = new TTree( "T_Gauss0", "T_Gauss0" );
  T_Gauss0->SetDirectory( 0 );
  double* data_Gauss0 = new double[layer.size()];
  for ( unsigned int l = 0; l < layer.size(); l++ )
    T_Gauss0->Branch( Form( "data_Gauss0_%s", layer[l].c_str() ), &data_Gauss0[l],
                      Form( "data_Gauss0_%s/D", layer[l].c_str() ) );

  TTree* T_Gauss = new TTree( "T_Gauss", "T_Gauss" );
  T_Gauss->SetDirectory( 0 );
  double* data_Gauss = new double[layer.size()];
  double* data_PCA   = new double[layer.size()];
  for ( unsigned int l = 0; l < layer.size(); l++ ) {
    T_Gauss->Branch( Form( "data_Gauss_%s", layer[l].c_str() ), &data_Gauss[l],
                     Form( "data_Gauss_%s/D", layer[l].c_str() ) );
    T_Gauss->Branch( Form( "data_PCA_comp%i", l ), &data_PCA[l], Form( "data_PCA_comp%i/D", l ) );
  }

  cout << "--- Uniformization/Gaussianization" << endl;
  for ( int event = 0; event < read_inputTree->GetEntries(); event++ ) {
    read_inputTree->GetEntry( event );
    bool pass_eta = 0;
    if ( !m_apply_etacut ) pass_eta = 1;
    if ( m_apply_etacut ) {
      double         E  = read_inputTree->GetVariable( "TruthE" );
      double         px = read_inputTree->GetVariable( "TruthPx" );
      double         py = read_inputTree->GetVariable( "TruthPy" );
      double         pz = read_inputTree->GetVariable( "TruthPz" );
      TLorentzVector tlv;
      tlv.SetPxPyPzE( px, py, pz, E );
      pass_eta = ( fabs( tlv.Eta() ) > m_cut_eta_low && fabs( tlv.Eta() ) < m_cut_eta_high );
    }
    if ( pass_eta ) {
      double total_e = read_inputTree->GetVariable( "total_cell_energy" );

      if ( fabs( total_e ) < 0.0001 ) continue;

      vector<double> e_layer;
      for ( unsigned int l = 0; l < layerNr.size(); l++ )
        e_layer.push_back( read_inputTree->GetVariable( Form( "cell_energy[%d]", layerNr[l] ) ) );

      /*
      int all_bad=1;
      for(unsigned int l=0;l<layerNr.size();l++)
      {
       if(fabs(e_layer[l])>0.0001) all_bad=0;
      }
      if(all_bad) continue;
      */

      // rescale to get rid of negative fractions:
      if ( m_dorescale ) {
        double e_positive_sum = 0.0;
        for ( unsigned int e = 0; e < layerNr.size(); e++ ) {
          if ( e_layer[e] > 0 )
            e_positive_sum += e_layer[e];
          else
            e_layer[e] = 0.0;
        }
        double rescale = 1.0;
        if ( e_positive_sum > 0 ) rescale = total_e / e_positive_sum;
        for ( unsigned int e = 0; e < e_layer.size(); e++ ) e_layer[e] *= rescale;
      }

      for ( unsigned int l = 0; l < layer.size(); l++ ) {
        double data = 0.;

        if ( l == layer.size() - 1 )
          data = total_e;
        else
          data = e_layer[l] / total_e;

        // cout<<"l "<<l<<" e_layer "<<e_layer[l]<<endl;

        // Uniformization
        // double cumulant = get_cumulant_random(data,cumul_data[l]);
        double cumulant = get_cumulant_random( randEngine, data, cumul_data[l] );

        // Gaussianization
        double maxErfInvArgRange = 0.99999999;
        double arg               = 2.0 * cumulant - 1.0;
        arg                      = TMath::Min( +maxErfInvArgRange, arg );
        arg                      = TMath::Max( -maxErfInvArgRange, arg );
        data_Gauss0[l]           = TMath::Pi() / 2.0 * TMath::ErfInverse( arg );

      } // for layers

      principal->AddRow( data_Gauss0 );
      T_Gauss0->Fill();

    } // pass eta

    if ( event % 2000 == 0 ) cout << event << " from " << read_inputTree->GetEntries() << " done " << endl;

  } // event loop

  cout << "--- MakePrincipals()" << endl;
  principal->MakePrincipals();

  cout << "--- PCA Results" << endl;
  principal->Print( "MSE" );

  TreeReader* read_T_Gauss0 = new TreeReader();
  read_T_Gauss0->SetTree( T_Gauss0 );

  // save both the input and the ouput of the PCA (output is just for debug)
  for ( int event = 0; event < read_T_Gauss0->GetEntries(); event++ ) {
    read_T_Gauss0->GetEntry( event );
    for ( unsigned int l = 0; l < layer.size(); l++ )
      data_Gauss[l] = read_T_Gauss0->GetVariable( Form( "data_Gauss0_%s", layer[l].c_str() ) );
    principal->X2P( data_Gauss, data_PCA );

    // for(unsigned int l=0;l<layer.size();l++)    cout<<"l "<<l<<" layer "<<layer[l]<<" Gauss "<<data_Gauss[l]<<" PCA
    // "<<data_PCA[l]<<endl;

    T_Gauss->Fill();
  }

  TFile* output = TFile::Open( m_outfilename.c_str(), "RECREATE" );
  output->Add( principal );
  output->Add( T_Gauss );
  output->Add( h_layer_input );
  output->Write();

  cout << "1st PCA is made. Output file: " << m_outfilename << endl;

  // cleanup
  delete read_inputTree;
  delete principal;
  delete T_Gauss;
  delete[] data_Gauss;
  delete[] data_PCA;

} // run

vector<TH1D*> TFCSMakeFirstPCA::get_cumul_histos( vector<string> layer_totE_name, vector<TH1D*> histos ) {

  cout << "in TFCSMakeFirstPCA::get_cumul_histos" << endl;

  vector<TH1D*> cumul;

  for ( unsigned int i = 0; i < histos.size(); i++ ) {
    TH1D* h_cumul = (TH1D*)histos[i]->Clone( Form( "h_cumul_%s", layer_totE_name[i].c_str() ) );
    for ( int b = 1; b <= h_cumul->GetNbinsX(); b++ ) {
      h_cumul->SetBinContent( b, histos[i]->Integral( 1, b ) );
      h_cumul->SetBinError( b, 0 );
    }
    cumul.push_back( h_cumul );
  }

  return cumul;
}

vector<int> TFCSMakeFirstPCA::get_relevantlayers( TreeReader* read_inputTree, double ecut ) {

  cout << "in TFCSMakeFirstPCA::get_relevantlayers" << endl;

  vector<int> layer_number;

  int NLAYER = 25;

  vector<double> sum_efraction;
  vector<double> sum_e;

  for ( int l = 0; l < NLAYER; l++ ) {
    sum_efraction.push_back( 0.0 );
    sum_e.push_back( 0.0 );
  }

  int good_events = 0;
  for ( int event = 0; event < read_inputTree->GetEntries(); event++ ) {
    read_inputTree->GetEntry( event );
    int  event_ok = 1;
    bool pass_eta = 0;
    if ( !m_apply_etacut ) pass_eta = 1;
    if ( m_apply_etacut ) {
      double         E  = read_inputTree->GetVariable( "TruthE" );
      double         px = read_inputTree->GetVariable( "TruthPx" );
      double         py = read_inputTree->GetVariable( "TruthPy" );
      double         pz = read_inputTree->GetVariable( "TruthPz" );
      TLorentzVector tlv;
      tlv.SetPxPyPzE( px, py, pz, E );
      pass_eta = ( fabs( tlv.Eta() ) > m_cut_eta_low && fabs( tlv.Eta() ) < m_cut_eta_high );
    }
    if ( pass_eta ) {
      double total_e = read_inputTree->GetVariable( "total_cell_energy" );
      if ( total_e > 0.0001 ) {
        for ( int l = 0; l < NLAYER; l++ ) {
          double efraction = read_inputTree->GetVariable( Form( "cell_energy[%d]", l ) ) / total_e;
          if ( efraction / total_e > 1 || efraction / total_e < 0 ) event_ok = 0;
        }
        if ( event_ok ) {
          good_events++;
          for ( int l = 0; l < NLAYER; l++ ) {
            double eval = read_inputTree->GetVariable( Form( "cell_energy[%i]", l ) );
            sum_efraction[l] += eval / total_e;
            sum_e[l] += eval;
          }
        } // event is good
      }   // total_e is positive
    }     // pass-eta
    if ( event % 2000 == 0 ) cout << event << " from " << read_inputTree->GetEntries() << " done" << endl;
  }

  cout << "rel. layer" << endl;

  // first criteria: (the "standard" one)
  // if average energy fraction of a layer is above the threshold, it counts as relevant

  double adc_to_energy_conversion[24]; // in MeV

  adc_to_energy_conversion[0]  = 3.5;  //"PreB";
  adc_to_energy_conversion[1]  = 1.0;  //"EMB1";
  adc_to_energy_conversion[2]  = 3.5;  //"EMB2";
  adc_to_energy_conversion[3]  = 5.0;  //"EMB3";
  adc_to_energy_conversion[4]  = 10.0; //"PreE";
  adc_to_energy_conversion[5]  = 1.5;  //"EME1";
  adc_to_energy_conversion[6]  = 5.0;  //"EME2";
  adc_to_energy_conversion[7]  = 4.0;  //"EME3";
  adc_to_energy_conversion[8]  = 55.0; //"HEC0";
  adc_to_energy_conversion[9]  = 55.0; //"HEC1";
  adc_to_energy_conversion[10] = 55.0; //"HEC2";
  adc_to_energy_conversion[11] = 55.0; //"HEC3";

  adc_to_energy_conversion[12] = 6.0; //"TileB0";
  adc_to_energy_conversion[13] = 6.0; //"TileB1";
  adc_to_energy_conversion[14] = 6.0; //"TileB2";
  adc_to_energy_conversion[15] = 6.0; //"TileGap1";
  adc_to_energy_conversion[16] = 6.0; //"TileGap2";
  adc_to_energy_conversion[17] = 6.0; //"TileGap3";
  adc_to_energy_conversion[18] = 6.0; //"TileExt0";
  adc_to_energy_conversion[19] = 6.0; //"TileExt1";
  adc_to_energy_conversion[20] = 6.0; //"TileExt2";

  adc_to_energy_conversion[21] = 43.0; //"FCAL0";
  adc_to_energy_conversion[22] = 75.0; //"FCAL1";
  adc_to_energy_conversion[23] = 85.0; //"FCAL2";

  for ( int l = 0; l < NLAYER; l++ ) {
    if ( sum_efraction[l] / (double)good_events >= ecut ) {
      if ( ( m_use_absolute_layercut && sum_e[l] / (double)good_events >= adc_to_energy_conversion[l] ) ||
           !m_use_absolute_layercut ) {
        layer_number.push_back( l );
        cout << "Layer " << l << " is relevant! <sum_efraction>= " << sum_efraction[l] / (double)good_events
             << " <sum_e> " << sum_e[l] / (double)good_events << endl;
      }
    }
  }

  for ( unsigned int k = 0; k < layer_number.size(); k++ ) cout << "Relevant " << layer_number[k] << endl;

  return layer_number;
}

vector<TH1D*> TFCSMakeFirstPCA::get_G4_histos_from_tree( vector<int> layer_number, TreeReader* read_inputTree ) {

  cout << "in TFCSMakeFirstPCA::get_G4_histos_from_tree" << endl;

  double max_e = 0;
  double min_e = 100000000;

  cout << "layer_number size " << layer_number.size() << endl;

  for ( int event = 0; event < read_inputTree->GetEntries(); event++ ) {
    read_inputTree->GetEntry( event );
    bool pass_eta = 0;
    if ( !m_apply_etacut ) pass_eta = 1;
    if ( m_apply_etacut ) {
      TLorentzVector tlv;
      tlv.SetPxPyPzE( read_inputTree->GetVariable( "TruthPx" ), read_inputTree->GetVariable( "TruthPy" ),
                      read_inputTree->GetVariable( "TruthPz" ), read_inputTree->GetVariable( "TruthE" ) );
      pass_eta = ( fabs( tlv.Eta() ) > m_cut_eta_low && fabs( tlv.Eta() ) < m_cut_eta_high );
    }
    if ( pass_eta ) {
      double total_e = read_inputTree->GetVariable( "total_cell_energy" );
      // if(fabs(total_e)<0.0001) continue;
      vector<double> e_layer;
      for ( unsigned int l = 0; l < layer_number.size(); l++ ) {
        e_layer.push_back( read_inputTree->GetVariable( Form( "cell_energy[%d]", layer_number[l] ) ) );
      }
      /*
      int all_bad=1;
      for(unsigned int l=0;l<layer_number.size();l++)
      {
       if(fabs(e_layer[l])>0.0001) all_bad=0;
      }
      if(all_bad) continue;
      */
      if ( total_e > max_e ) max_e = total_e;
      if ( total_e < min_e ) min_e = total_e;
    }
  } // 1st event loop

  cout << "min_e " << min_e << " max_e " << max_e << endl;

  vector<TH1D*> h_data;
  cout << "init data histos" << endl;
  for ( unsigned int l = 0; l < layer_number.size(); l++ ) {
    TH1D* hist = new TH1D( Form( "h_data_layer%i", l ), Form( "h_data_layer%i", l ), m_numberfinebins, 0, 1 );
    hist->Sumw2();
    h_data.push_back( hist );
  }

  TH1D* h_data_total = new TH1D( "h_data_totalE", "h_data_totalE", m_numberfinebins, min_e, max_e );
  h_data.push_back( h_data_total );

  cout << "fill data histos" << endl;

  for ( int event = 0; event < read_inputTree->GetEntries(); event++ ) {
    read_inputTree->GetEntry( event );

    bool pass_eta = 0;
    if ( !m_apply_etacut ) pass_eta = 1;
    if ( m_apply_etacut ) {
      TLorentzVector tlv;
      tlv.SetPxPyPzE( read_inputTree->GetVariable( "TruthPx" ), read_inputTree->GetVariable( "TruthPy" ),
                      read_inputTree->GetVariable( "TruthPz" ), read_inputTree->GetVariable( "TruthE" ) );
      pass_eta = ( fabs( tlv.Eta() ) > m_cut_eta_low && fabs( tlv.Eta() ) < m_cut_eta_high );
    }
    if ( pass_eta ) {
      double total_e = read_inputTree->GetVariable( "total_cell_energy" );
      // if(fabs(total_e)<0.0001) continue;
      vector<double> e_layer;
      for ( unsigned int l = 0; l < layer_number.size(); l++ )
        e_layer.push_back( read_inputTree->GetVariable( Form( "cell_energy[%d]", layer_number[l] ) ) );
      /*
      int all_bad=1;
      for(unsigned int l=0;l<layer_number.size();l++)
      {
       if(fabs(e_layer[l])>0.0001) all_bad=0;
      }
      if(all_bad) continue;
      */

      // rescale to get rid of negative fractions:
      if ( m_dorescale ) {
        double e_positive_sum = 0.0;
        for ( unsigned int e = 0; e < layer_number.size(); e++ ) {
          if ( e_layer[e] > 0 )
            e_positive_sum += e_layer[e];
          else
            e_layer[e] = 0.0;
        }
        double rescale = 1.0;
        if ( e_positive_sum > 0 ) rescale = total_e / e_positive_sum;
        for ( unsigned int e = 0; e < e_layer.size(); e++ ) e_layer[e] *= rescale;

        h_data[h_data.size() - 1]->Fill( total_e );
        for ( unsigned int l = 0; l < layer_number.size(); l++ ) h_data[l]->Fill( e_layer[l] / total_e );
      }
    } // pass eta
    if ( event % 2000 == 0 ) cout << event << " from " << read_inputTree->GetEntries() << " done" << endl;
  } // for event

  for ( unsigned int l = 0; l < h_data.size(); l++ ) h_data[l]->Scale( 1.0 / h_data[l]->Integral() );

  return h_data;
}

double TFCSMakeFirstPCA::get_cumulant( double x, TH1D* h ) {

  int bin = h->FindBin( x );
  return h->GetBinContent( bin );
}

// double TFCSMakeFirstPCA::get_cumulant_random(double x, TH1D* h)
double TFCSMakeFirstPCA::get_cumulant_random( CLHEP::HepRandomEngine* randEngine, double x, TH1D* h ) {

  int bin = h->FindBin( x );

  double content        = h->GetBinContent( bin );
  double before_content = h->GetBinContent( bin - 1 );

  // TRandom3 ran(0);
  // double cumulant=ran.Uniform(before_content,content);

  double cumulant = CLHEP::RandFlat::shoot( randEngine, before_content, content );

  return cumulant;
}
