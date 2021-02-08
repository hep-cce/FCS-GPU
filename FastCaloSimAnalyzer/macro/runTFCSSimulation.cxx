/*
  Copyright (C) 2002-2019 CERN for the benefit of the ATLAS collaboration
*/

#include "TFile.h"
#include "TCanvas.h"
#include "TChain.h"
#include "TPaveText.h"
#include <docopt/docopt.h>
#include "FastCaloSimAnalyzer/TFCSShapeValidation.h"
#include "FastCaloSimAnalyzer/TFCSValidationEnergyAndHits.h"
#include "TFCSSampleDiscovery.h"
#include <chrono>

#ifdef USE_KOKKOS
#  include <Kokkos_Core.hpp>
#endif

using namespace std;

float       init_eta;
std::string prefix_E_eta;
std::string prefix_E_eta_title;
std::string prefixlayer;
std::string prefixEbin;
std::string prefixall;
std::string prefixlayer_title;
std::string prefixEbin_title;
std::string prefixall_title;

TFile* fout{};

static const char* USAGE =
    R"(Run toy simulation to validate the shape parametrization

Usage:
  runTFCSSimulation [--pdgId <pdgId>] [-s <seed> | --seed <seed>] [-o <file> | --output <file>] [--energy <energy>] [--etaMin <etaMin>] [-l <layer> | --layer <layer>] [--nEvents <nEvents>] [--firstEvent <firstEvent>] [--pcabin <pcabin>] [--debug <debug>] [--png]
  runTFCSSimulation (-h | --help)

Options:
  -h --help                    Show help screen.
  --pdgId <pdgId>              Particle ID [default: 11].
  -s <seed>, --seed <seed>     Random seed [default: 42].
  --energy <energy>            Input sample energy in MeV. Should match energy point on the grid. [default: 65536].
  --etaMin <etaMin>            Minimum eta of the input sample. Should match eta point on the grid. [default: 0.2].
  -o <file>, --output <file>   Output plot file name [default: Simulation.root].
  -l <layer>, --layer <layer>  Layer to analyze [default: 2].
  --nEvents <nEvents>          Number of events to run over with. All events will be used if nEvents<=0 [default: -1].
  --firstEvent <firstEvent>    Run will start from this event [default: 0].
  --pcabin <pcabin>            Select over which pcabin to run. pcabin=-1 runs over all bins and over each individual bin. pcabin=-2 runs only over all bins [default: -1].
  --debug <debug>              Set debug level to print debug messages [default: 0].
  --png                        Save all the histograms in .png images.
)";

void Draw_1Dhist( TH1* hist1, double ymin = 0, double ymax = 0, bool logy = false, TString name = "",
                  TString title = "", TCanvas* c = 0, bool png = false ) {
  if ( name == "" ) {
    name  = hist1->GetName();
    title = hist1->GetTitle();
  }

  if ( c ) {
    c->SetName( name );
    c->SetTitle( title );
  } else {
    c = new TCanvas( name, title );
  }

  double min1, max1, rmin1, rmax1;
  TFCSAnalyzerBase::autozoom( hist1, min1, max1, rmin1, rmax1 );

  TPaveText* pt = new TPaveText( 0.9, 0.5, 1.0, 0.9, "NDC" );
  pt->SetFillColor( 10 );
  pt->SetBorderSize( 1 );
  TText* t1;

  TH1D* newhist1 = TFCSAnalyzerBase::refill( hist1, min1, max1, rmin1, rmax1 );
  newhist1->SetLineColor( 1 );
  newhist1->SetTitle( title );
  newhist1->SetStats( false );
  if ( ymin != 0 || ymax != 0 ) {
    newhist1->SetMaximum( ymax );
    newhist1->SetMinimum( ymin );
  }
  newhist1->Draw( "EL" );
  newhist1->Write();

  t1 = pt->AddText( "Mean:" );
  t1->SetTextFont( 62 );
  t1 = pt->AddText( Form( "%5.3f#pm%5.3f", hist1->GetMean(), hist1->GetMeanError() ) );
  t1->SetTextFont( 42 );

  t1 = pt->AddText( "RMS:" );
  t1->SetTextFont( 62 );
  t1 = pt->AddText( Form( "%5.3f#pm%5.3f", hist1->GetRMS(), hist1->GetRMSError() ) );
  t1->SetTextFont( 42 );

  t1 = pt->AddText( "Skewness:" );
  t1->SetTextFont( 62 );
  t1 = pt->AddText( Form( "%5.3f#pm%5.3f", hist1->GetSkewness(), hist1->GetSkewness( 11 ) ) );
  t1->SetTextFont( 42 );

  pt->Draw();
  c->SetLogy( logy );
  if ( png ) { c->SaveAs( ".png" ); }
  return;
}

void FillEnergyHistos( TH1** hist_E, TFCSShapeValidation* analyze, int analyze_pcabin, TFCSSimulationRun& val1 ) {
  hist_E[24] =
      analyze->InitTH1( prefixEbin + "E_over_Ekintrue_" + val1.GetName(), "1D", 840, 0, 2.0, "E/Ekin(true)", "#" );
  hist_E[24]->SetTitle( val1.GetTitle() );
  for ( int i = 0; i < 24; ++i ) {
    hist_E[i] = analyze->InitTH1( prefixEbin + Form( "E%02d_over_E_", i ) + val1.GetName(), "1D", 840, 0, 1.0,
                                  Form( "E%d/E", i ), "#" );
    hist_E[i]->SetTitle( val1.GetTitle() );
  }
  for ( size_t ievent = 0; ievent < val1.simul().size(); ++ievent ) {
    const TFCSSimulationState& simul_val1 = val1.simul()[ievent];
    if ( simul_val1.Ebin() != analyze_pcabin && analyze_pcabin >= 0 ) continue;
    for ( int i = 0; i < 24; ++i ) { TFCSAnalyzerBase::Fill( hist_E[i], simul_val1.E( i ) / simul_val1.E(), 1 ); }
    TFCSAnalyzerBase::Fill( hist_E[24], simul_val1.E() / analyze->get_truthTLV( ievent ).Ekin(), 1 );
  }
}

void Energy_histograms( TFCSShapeValidation* analyze, int analyze_pcabin, TFCSSimulationRun& val2,
                        TString basename = "", bool png = false ) {
  TH1* hist_E_val2[25];
  FillEnergyHistos( hist_E_val2, analyze, analyze_pcabin, val2 );
  for ( int i = 0; i < 25; ++i ) {
    if ( hist_E_val2[i]->GetMean() > 0 ) {
      TString name  = basename + "_" + Form( "cs%02d_", i ) + prefixEbin;
      TString title = basename + ": " + Form( "sample=%d, ", i ) + prefixEbin_title;
      if ( i == 24 ) {
        name  = basename + "_" + Form( "total_" ) + prefixEbin;
        title = basename + ": " + Form( "E/E_{true}, " ) + prefixEbin_title;
      }
      Draw_1Dhist( hist_E_val2[i], 0, 0, false, name, title, 0, png );
    }
  }
}

void set_prefix( int analyze_layer, int analyze_pcabin ) {
  prefixlayer       = prefix_E_eta + Form( "cs%02d_", analyze_layer );
  prefixlayer_title = prefix_E_eta_title + Form( ", sample=%d", analyze_layer );
  if ( analyze_pcabin >= 0 ) {
    prefixall        = prefix_E_eta + Form( "cs%02d_pca%d_", analyze_layer, analyze_pcabin );
    prefixall_title  = prefix_E_eta_title + Form( ", sample=%d, pca=%d", analyze_layer, analyze_pcabin );
    prefixEbin       = prefix_E_eta + Form( "pca%d_", analyze_pcabin );
    prefixEbin_title = prefix_E_eta_title + Form( ", pca=%d", analyze_pcabin );
  } else {
    prefixall        = prefix_E_eta + Form( "cs%02d_allpca_", analyze_layer );
    prefixall_title  = prefix_E_eta_title + Form( ", sample=%d, all pca", analyze_layer );
    prefixEbin       = prefix_E_eta + Form( "allpca_" );
    prefixEbin_title = prefix_E_eta_title + Form( ", all pca" );
  }
}

int runTFCSSimulation( int pdgid = 22, int int_E = 65536, double etamin = 0.2, int analyze_layer = 2,
                       const std::string& plotfilename = "Simulation.root", long seed = 42, int nEvents = -1,
                       int firstEvent = 0, int selectPCAbin = -1, int debug = 0, bool png = false ) {
  auto t0 = std::chrono::system_clock::now();

  TFCSParametrizationBase* fullchain     = nullptr;
  std::string              paramName     = TFCSSampleDiscovery::getParametrizationName();
  auto                     t00_A         = std::chrono::system_clock::now();
  auto                     fullchainfile = std::unique_ptr<TFile>( TFile::Open( paramName.c_str() ) );
  std::cout << "Parametrization File: '" << paramName << "'" << std::endl;
  if ( !fullchainfile ) {
    std::cerr << "Error: Could not open file '" << paramName << "'" << std::endl;
    return 1;
  }
  auto t0_A = std::chrono::system_clock::now();
#ifdef FCS_DEBUG
  fullchainfile->ls();
#endif
  auto t01  = std::chrono::system_clock::now();
  fullchain = dynamic_cast<TFCSParametrizationBase*>( fullchainfile->Get( "SelPDGID" ) );
  fullchainfile->Close();
  auto   t1            = std::chrono::system_clock::now();
  double etamax        = etamin + 0.05;
  init_eta             = etamin + 0.025;
  std::string particle = "";
  if ( pdgid == 22 ) particle = "photon";
  if ( pdgid == 211 ) particle = "pion";
  if ( pdgid == 11 ) particle = "electron";
  std::string energy = Form( "E%d", int_E );
  std::string eta    = Form( "eta%03d_%03d", TMath::Nint( etamin * 100 ), TMath::Nint( etamax * 100 ) );
  prefix_E_eta       = ( particle + "_" + energy + "_" + eta + "_" ).c_str();
  prefix_E_eta_title = particle + Form( ", E=%d MeV, %4.2f<|#eta|<%4.2f", int_E, etamin, etamax );
  std::string energy_label( energy );
  energy_label.erase( 0, 1 );
  int part_energy = stoi( energy_label );
  std::cout << " energy = " << part_energy << std::endl;
  std::string eta_label( eta );
  eta_label.erase( 0, 3 );
  std::cout << " eta_label = " << eta_label << std::endl;
  std::string     etamin_label = eta_label.substr( 0, eta_label.find( "_" ) );
  std::string     etamax_label = eta_label.substr( 4, eta_label.find( "_" ) );
  auto            t01_A        = std::chrono::system_clock::now();
  auto            sample       = std::make_unique<TFCSSampleDiscovery>();
  int             dsid         = sample->findDSID( pdgid, int_E, etamin * 100, 0 ).dsid;
  FCS::SampleInfo sampleInfo   = sample->findSample( dsid );
  TString         inputSample  = sampleInfo.location;
  TString         shapefile    = sample->getShapeName( dsid );
  TString         energyfile   = sample->getSecondPCAName( dsid );
  TString         pcaSample    = sample->getFirstPCAAppName( dsid );
  TString         avgSample    = sample->getAvgSimShapeName( dsid );
  set_prefix( analyze_layer, -1 );
  auto t01_B = std::chrono::system_clock::now();
#if defined( __linux__ )
  std::cout << "* Running on linux system " << std::endl;
#endif
  std::cout << dsid << "\t" << inputSample << std::endl;
  TChain* inputChain = new TChain( "FCS_ParametrizationInput" );
  if ( inputChain->Add( inputSample, -1 ) == 0 ) {
    std::cerr << "Error: Could not open file '" << inputSample << "'" << std::endl;
    return 1;
  }
  int nentries = inputChain->GetEntries();
  if ( nEvents <= 0 ) {
    if ( firstEvent >= 0 )
      nEvents = nentries;
    else
      nEvents = nentries;
  } else {
    if ( firstEvent >= 0 )
      nEvents = std::max( 0, std::min( nentries, nEvents + firstEvent ) );
    else
      nEvents = std::max( 0, std::min( nentries, nEvents ) );
  }

  std::cout << " * Prepare to run on: " << inputSample << " with entries = " << nentries << std::endl;
  std::cout << " * Running over " << nEvents << " events." << std::endl;
  std::cout << " *   1stPCA file: " << pcaSample << std::endl;
  std::cout << " *   AvgShape file: " << avgSample << std::endl;

  auto t01_C = std::chrono::system_clock::now();
  //////////////////////////////////////////////////////////
  ///// Creat validation steering
  //////////////////////////////////////////////////////////
  TFCSShapeValidation* analyze = new TFCSShapeValidation( inputChain, analyze_layer, seed );
  auto                 t2A     = std::chrono::system_clock::now();
  analyze->set_IsNewSample( true );
  analyze->set_Nentries( nEvents );
  analyze->set_Debug( debug );
  analyze->set_firstevent( firstEvent );

  //////////////////////////////////////////////////////////
  ///// Chain to simulate energy from PCA and the shape from a histogram
  //////////////////////////////////////////////////////////
  int ind_fullchain = -1;
  if ( fullchain ) {
    ind_fullchain = analyze->add_validation( "AllSim", "Energy+shape sim", fullchain );
    std::cout << "=============================" << std::endl;
  }

  auto t2 = std::chrono::system_clock::now();

  //////////////////////////////////////////////////////////
  ///// Run over events
  //////////////////////////////////////////////////////////
  analyze->LoopEvents( -1 );

  auto t3 = std::chrono::system_clock::now();

  if ( plotfilename != "" ) {
    fout = TFile::Open( plotfilename.c_str(), "recreate" );
    if ( !fout ) {
      std::cerr << "Error: Could not create file '" << plotfilename << "'" << std::endl;
      return 1;
    }
    fout->cd();
  }
  int     npcabins    = 5;
  int     ibin        = 1;
  TString nameenergy  = "Energy";
  int     firstpcabin = -1;
  int     lastpcabin  = npcabins;
  if ( selectPCAbin == -2 ) lastpcabin = -1;
  if ( selectPCAbin > 0 ) {
    firstpcabin = selectPCAbin;
    lastpcabin  = selectPCAbin;
  }
  for ( int i = firstpcabin; i <= lastpcabin; ++i ) {
    if ( i == 0 ) continue;
    int analyze_pcabin = i;
    set_prefix( analyze_layer, analyze_pcabin );

    if ( ind_fullchain >= 0 ) {
      TFCSSimulationRun& val2 = analyze->validations()[ind_fullchain];
      Energy_histograms( analyze, analyze_pcabin, val2, nameenergy, png );
    }
    ++ibin;
  }
  auto                          t4   = std::chrono::system_clock::now();
  std::chrono::duration<double> diff = t2 - t0;
  std::cout << "Time before loop:" << diff.count() << " s" << std::endl;
  diff = t00_A - t0;
  // std::cout<<"Time of getParamName:"<<diff.count()<<" s" <<std::endl ;
  diff = t0_A - t0;
  std::cout << "Time of Tfile::Open():" << diff.count() << " s" << std::endl;
  diff = t1 - t01;
  std::cout << "Time of fullchain->Get():" << diff.count() << " s" << std::endl;
  diff = t01_A - t1;
  // std::cout<<"Time of NamePrefixHandle:"<<diff.count()<<" s" <<std::endl ;
  diff = t01 - t0_A;
  // std::cout<<"Time of ls:"<<diff.count()<<" s" <<std::endl ;
  diff = t01_B - t01_A;
  // std::cout<<"Time of SampleDiscover:"<<diff.count()<<" s" <<std::endl ;
  diff = t01_C - t01_B;
  // std::cout<<"Time of Add Input:"<<diff.count()<<" s" <<std::endl ;
  diff = t2A - t01_C;
  std::cout << "Time of New Validation:" << diff.count() << " s" << std::endl;
  // diff = t2-t2A ;
  // std::cout<<"Time of Add_Validation:"<<diff.count()<<" s" <<std::endl ;
  diff = t3 - t2;
  std::cout << "Time in funtion loopevent():" << diff.count() << " s" << std::endl;
  diff = t4 - t3;
  std::cout << "Time after loopevent:" << diff.count() << " s" << std::endl;
  diff = t4 - t0;
  std::cout << "Time of total run:" << diff.count() << " s" << std::endl;
  return 0;
}

int main( int argc, char** argv ) {

  std::map<std::string, docopt::value> args = docopt::docopt( USAGE, {argv + 1, argv + argc}, true );

  int         pdgId        = args["--pdgId"].asLong();
  int         energy       = args["--energy"].asLong();
  double      etamin       = std::stof( args["--etaMin"].asString() );
  long        seed         = args["--seed"].asLong();
  std::string output       = args["--output"].asString();
  int         layer        = args["--layer"].asLong();
  int         nEvents      = args["--nEvents"].asLong();
  int         firstEvent   = args["--firstEvent"].asLong();
  int         selectPCAbin = args["--pcabin"].asLong();
  int         debug        = args["--debug"].asLong();
  bool        png          = args["--png"].asBool();
#ifdef USE_KOKKOS
  Kokkos::initialize( argc, argv );
#endif
  int ret =
      runTFCSSimulation( pdgId, energy, etamin, layer, output, seed, nEvents, firstEvent, selectPCAbin, debug, png );

#ifdef USE_KOKKOS
  Kokkos::finalize();
#endif
  return ret;
}
