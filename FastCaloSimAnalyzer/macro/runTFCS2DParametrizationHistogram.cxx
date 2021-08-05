/*
  Copyright (C) 2002-2019 CERN for the benefit of the ATLAS collaboration
*/

#include <algorithm>
#include <iostream>
#include <stdlib.h>
#include <string>
#include <tuple>
#include <vector>

#include <docopt/docopt.h>

#include <TCanvas.h>
#include <TChain.h>
#include <TFile.h>
#include <TH1.h>
#include <TH2.h>
#include <TROOT.h>

#include <CLHEP/Random/RandomEngine.h>
#include <CLHEP/Random/TRandomEngine.h>

#include <ISF_FastCaloSimEvent/TFCSCenterPositionCalculation.h>
#include <ISF_FastCaloSimEvent/TFCSParametrizationChain.h>

#include "TFCSApplyFirstPCA.h"
#include "TFCSMakeFirstPCA.h"

#include "FastCaloSimAnalyzer/TFCSAnalyzerBase.h"
#include "FastCaloSimAnalyzer/TFCSShapeValidation.h"
#include "FastCaloSimAnalyzer/TFCSValidationEnergyAndHits.h"
#include "FastCaloSimAnalyzer/TFCSValidationHitSpy.h"

#include "TFCSSampleDiscovery.h"


TH1F* zoomHisto(TH1* h_in)
{

  double min = -999.;
  double max = 999.;

  double rmin = -999.;
  double rmax = 999.;

  TFCSAnalyzerBase::autozoom( h_in, min, max, rmin, rmax );

  int Nbins;
  int bins = 0;
    for (int b = h_in->FindBin(min); b <= h_in->FindBin(max); b++)
        bins++;
  Nbins = bins;

  int start = h_in->FindBin( min ) - 1;

  TH1F* h_out = new TH1F( h_in->GetName() + TString( "_zoom" ), h_in->GetTitle(), Nbins, rmin, rmax );
  h_out->SetXTitle( h_in->GetXaxis()->GetTitle() );
  h_out->SetYTitle( h_in->GetYaxis()->GetTitle() );
    for (int b = 1; b <= h_out->GetNbinsX(); b++)
    {
    h_out->SetBinContent( b, h_in->GetBinContent( start + b ) );
    h_out->SetBinError( b, h_in->GetBinError( start + b ) );
  }

  return h_out;
}


TH2* Create2DHistogram(TH2* h, float energy_cutoff)
{

  TH1F* h1 = (TH1F*)h->ProjectionY();

  float ymin      = h->GetYaxis()->GetBinLowEdge( 1 );
  float ymax      = TFCSAnalyzerBase::GetBinUpEdge( h1, energy_cutoff );
  float binwidthx = h->GetYaxis()->GetBinWidth( 1 );
  int   nbinsy    = (int)( ( ymax - ymin ) / binwidthx );

  float xmin   = h->GetXaxis()->GetXmin();
  float xmax   = h->GetXaxis()->GetXmax();
  int   nbinsx = h->GetXaxis()->GetNbins();


    TH2* h2 = new TH2F(h->GetName() + TString("2D"), h->GetTitle() + TString("2D"), nbinsx, xmin, xmax, nbinsy, ymin, ymax);


  for ( auto j = 0; j <= h2->GetNbinsY(); ++j ) {
        for (auto i = 0; i <= h2->GetNbinsX(); ++i) {
            h2->SetBinContent(i, j, h->GetBinContent(i, j));
        }
  }

  float avg = h2->Integral( 1, nbinsx, 1, 1 ) / nbinsx;


    for (int i = 1; i < nbinsx + 1; i++)
        h2->SetBinContent(i, 1, avg);


  float integral = h2->Integral();
  if ( integral > 0 ) h2->Scale( 1 / integral );

  return h2;
}

void CreatePolarPlot(TH2F* h, std::string outDir)
{

  gROOT->SetBatch( 1 );

  system( ( "mkdir -p " + outDir ).c_str() );

  std::string title  = h->GetTitle();
  std::string xlabel = "x [mm]";
  std::string ylabel = "y [mm]";
  std::string zlabel = "Energy normalized to unity";

  TCanvas* c = TFCSAnalyzerBase::PlotPolar( h, title.c_str(), xlabel, ylabel, zlabel, 1 );

  std::string outfile = outDir + title;

  c->SaveAs( ( outfile + ".png" ).c_str() );

  delete c;
}


void runTFCS2DParametrizationHistogram(int dsid,
                                       int dsid_zv0,
                                       std::string sampleData,
                                       std::string topDir,
				                               std::string version,
                                       float energy_cutoff,
                                       std::string topPlotDir,
                                       bool do2DParam,
                                       bool isPhiSymmetry,
                                       bool doMeanRz,
                                       bool useMeanRz,
                                       bool doZVertexStudies,
                                       long seed,
	                                     int nEvents,
                        				       int firstEvent,
                        				       int debug)
{

  system( ( "mkdir -p " + topDir ).c_str() );

  // Create random engine
  CLHEP::TRandomEngine* randEngine = new CLHEP::TRandomEngine();
  randEngine->setSeed( seed );

  /////////////////////////////
  // read smaple information
  // based on DSID
  //////////////////////////

  if ( dsid_zv0 < 0 ) dsid_zv0 = dsid;

  auto            sample     = std::make_unique<TFCSSampleDiscovery>();
  FCS::SampleInfo sampleInfo = sample->findSample( dsid, sampleData );

  std::string input     = sampleInfo.location;
  std::string baselabel = sampleInfo.label;
  int         pdgid     = sampleInfo.pdgId;
  int         energy    = sampleInfo.energy;
  float       etamin    = sampleInfo.etaMin;
  float       etamax    = sampleInfo.etaMax;
  int         zv        = sampleInfo.zVertex;

  FCS::SampleInfo sample_zv0 = sample->findSample( dsid_zv0, sampleData );

  if ( pdgid == 211 ) energy_cutoff = 0.995; /// consider only 99.95% for pions

  std::cout << " *************************** " << std::endl;
  std::cout << " DSID : " << dsid << std::endl;
  std::cout << " location: " << input << std::endl;
  std::cout << " base name:  " << baselabel << std::endl;
  std::cout << " pdgID: " << pdgid << std::endl;
  std::cout << " energy (MeV) : " << energy << std::endl;
  std::cout << " eta min, max : " << etamin << " , " << etamax << std::endl;
  std::cout << " z vertex : " << zv << std::endl;
  std::cout << "*********************************" << std::endl;

  /////////////////////////////////////////
  // form names for ouput files and directories
  ///////////////////////////////////////////

  TString inputSample( Form( "%s", input.c_str() ) );
  TString pcaSample( Form( "%s%s.firstPCA.%s.root", topDir.c_str(), baselabel.c_str(), version.c_str() ) );
  TString shapeSample( Form( "%s%s.shapepara.%s.root", topDir.c_str(), baselabel.c_str(), version.c_str() ) );
  TString extrapolSample( Form( "%s%s.extrapol.%s.root", topDir.c_str(), baselabel.c_str(), version.c_str() ) );
  TString zvertexSample( Form( "%s%s.zvertex.%s.root", topDir.c_str(), baselabel.c_str(), version.c_str() ) );
  TString plotDir( Form( "%s/%s.plots.%s/", topPlotDir.c_str(), baselabel.c_str(), version.c_str() ) );

  TString pcaAppSample = pcaSample;
  pcaAppSample.ReplaceAll( "firstPCA", "firstPCA_App" );

  /////////////////////////////////////////
  // read input sample and create first pca
  ///////////////////////////////////////////
  TChain* inputChain = new TChain( "FCS_ParametrizationInput" );
  inputChain->Add( sample_zv0.location.c_str() );

  int nentries = inputChain->GetEntries();

  std::cout << " * Prepare to run on: " << inputSample << " with entries = " << nentries << std::endl;

  TFCSMakeFirstPCA* myfirstPCA = new TFCSMakeFirstPCA( inputChain, pcaSample.Data() );
  myfirstPCA->set_cumulativehistobins( 5000 );
  myfirstPCA->use_absolute_layercut( 1 ); /// apply ADC to MeV conversion
  myfirstPCA->run( randEngine );
  delete myfirstPCA;
  std::cout << "TFCSMakeFirstPCA done" << std::endl;

  delete inputChain;
  inputChain = new TChain( "FCS_ParametrizationInput" );
  inputChain->Add( inputSample );

  int npca1 = 5;
  int npca2 = 1;

  // pcaSample = Form("../../EnergyParametrization/scripts/output/ds%i.FirstPCA.ver01.root",dsid_zv0);

  TFCSApplyFirstPCA* myfirstPCA_App = new TFCSApplyFirstPCA( pcaSample.Data() );
  myfirstPCA_App->set_pcabinning( npca1, npca2 );
  myfirstPCA_App->init();
  myfirstPCA_App->run_over_chain( randEngine, inputChain, pcaAppSample.Data() );
  delete myfirstPCA_App;
  std::cout << "TFCSApplyFirstPCA done" << std::endl;

  // -------------------------------------------------------

  TChain* pcaChain = new TChain( "tree_1stPCA" );
  pcaChain->Add( pcaAppSample );
  inputChain->AddFriend( "tree_1stPCA" );

  /////////////////////////////////////// ///
  // get relevant layers and no. of PCA bins
  // from the firstPCA
  ////////////////////////////////////////////

  TFile*           fpca = TFile::Open( pcaAppSample );
  std::vector<int> v_layer;

  TH2I* relevantLayers = (TH2I*)fpca->Get( "h_layer" );
  int   npca           = relevantLayers->GetNbinsX();
    for (int ibiny = 1; ibiny <= relevantLayers->GetNbinsY(); ibiny++ )
    {
    if ( relevantLayers->GetBinContent( 1, ibiny ) == 1 ) v_layer.push_back( ibiny - 1 );
  }

  std::cout << " relevantLayers = ";
  for ( auto i : v_layer ) std::cout << i << " ";
  std::cout << "\n";

  //////////////////////////////////////////////////////////
  ///// Create validation steering
  //////////////////////////////////////////////////////////

  // v_layer.clear();
  // v_layer.push_back(2);
  // npca = 2;

  TFile*           fshape         = nullptr;
  TFile*           fextrapol      = nullptr;
  TFile*           fzvertex       = nullptr;
  TMatrixT<float>* tm_mean_weight = nullptr;

  if ( do2DParam ) fshape = new TFile( shapeSample, "recreate" );
  if ( doMeanRz ) {
    fextrapol      = new TFile( extrapolSample, "recreate" );
    tm_mean_weight = new TMatrixT<float>( 24, 6 );
  }
  if ( useMeanRz ) {
    TString name = Form( "%s%s.extrapol.%s.root", topDir.c_str(), sample_zv0.label.c_str(), version.c_str() );

    fextrapol      = new TFile( name, "read" );
    tm_mean_weight = (TMatrixT<float>*)fextrapol->Get( "tm_mean_weight" );
    std::cout << tm_mean_weight->GetName() << std::endl;
  }

    if (doZVertexStudies) {
        fzvertex = new TFile(zvertexSample, "recreate");
    }

  if ( nEvents <= 0 ) {
      if ( firstEvent >=0 ) nEvents=nentries;
      else nEvents=nentries;
  }
    else{
      if ( firstEvent >=0 ) nEvents=std::max( 0,std::min(nentries,nEvents+firstEvent) );
      else nEvents = std::max( 0,std::min(nentries,nEvents) );
    }


  for ( size_t ilayer = 0; ilayer < v_layer.size(); ilayer++ ) {

    std::vector<TH2*>  h_orig_hitEnergy_alpha_r( npca );
    std::vector<TH1F*> h_deltaEtaAveragedPerEvent( npca );
    std::vector<TH1F*> h_energyPerLayer( npca );
    std::vector<TH1F*> h_MeanEnergy( npca );
    std::vector<TH1F*> h_deltaEta( npca );
    std::vector<TH1F*> h_deltaPhi( npca );
    std::vector<TH1F*> h_deltaRt( npca );
    std::vector<TH1F*> h_deltaZ( npca );

    std::vector<TH1F*> h_hitenergy_r( npca );
    std::vector<TH1F*> h_hitenergy_z( npca );
    std::vector<TH1F*> h_hitenergy_weight( npca );

    std::vector<TH1F*> h_orig_mean_hitenergy_r( npca );
    std::vector<TH1F*> h_orig_mean_hitenergy_z( npca );
    std::vector<TH1F*> h_orig_mean_hitenergy_weight( npca );

    std::vector<TH2F*> h_Rz( npca );
    std::vector<TH2F*> h_Rz_outOfRange( npca );

    std::vector<TFCSParametrizationChain*>      RunInputHits( npca );
    std::vector<TFCSValidationEnergyAndHits*>   input_EnergyAndHits( npca );
	std::vector<TFCSCenterPositionCalculation*> centerPosCalc(npca); // Will decorate hits with center extrap positions
    std::vector<TFCSValidationHitSpy*> hitspy_orig( npca );

    int                 analyze_layer = v_layer.at( ilayer );
    TFCSShapeValidation analyze( inputChain, analyze_layer );
    analyze.set_IsNewSample( true );
    analyze.set_Nentries( nEvents );
    analyze.set_Debug( debug );
    analyze.set_firstevent( firstEvent );

    for ( int ipca = 1; ipca <= npca; ipca++ ) {

      int i              = ipca - 1;
      int analyze_pcabin = ipca;

      std::string prefixlayer = Form( "cs%d_", analyze_layer );
      std::string prefixall   = Form( "cs%d_pca%d_", analyze_layer, analyze_pcabin );
      std::string prefixEbin  = Form( "pca%d_", analyze_pcabin );

      std::cout << "=============================" << std::endl;

      //////////////////////////////////////////////////////////
      ///// Chain to read 2D alpha_radius in mm from the input file
      //////////////////////////////////////////////////////////


            RunInputHits[i] = new TFCSParametrizationChain("input_EnergyAndHits", "original energy and hits from input file");


            input_EnergyAndHits[i] = new TFCSValidationEnergyAndHits("input_EnergyAndHits", "original energy and hits from input file", &analyze);


      input_EnergyAndHits[i]->set_pdgid( pdgid );
      input_EnergyAndHits[i]->set_calosample( analyze_layer );
      input_EnergyAndHits[i]->set_Ekin_bin( analyze_pcabin );

      RunInputHits[i]->push_back( input_EnergyAndHits[i] );
      RunInputHits[i]->Print();

      centerPosCalc[i] = new TFCSCenterPositionCalculation( "CenterPosCalculation", "Center position calculation" );
      centerPosCalc[i]->set_calosample( analyze_layer );

	    if(useMeanRz) centerPosCalc[i]->setExtrapWeight( (*tm_mean_weight)[analyze_layer][ipca] );
	    else centerPosCalc[i]->setExtrapWeight(0.5);

      hitspy_orig[i] = new TFCSValidationHitSpy( "hitspy_2D_E_alpha_radius", "shape parametrization" );

      hitspy_orig[i]->set_calosample( analyze_layer );

      if ( doMeanRz ) RunInputHits[i]->push_back( hitspy_orig[i] ); // to call the simulate() method in HitSpy


      int binwidth = 5;
            if (analyze_layer == 1 or analyze_layer == 5)
                binwidth = 1;
      float ymin   = 0;
      float ymax   = 10000;
      int   nbinsy = (int)( ( ymax - ymin ) / binwidth );

      float rmin   = 0;
      float rmax   = 30000;
      int   nbinsr = (int)( ( rmax - rmin ) / binwidth );

      float zmin   = -10000;
      float zmax   = -zmin;
      int   nbinsz = (int)( ( zmax - zmin ) / binwidth );

      if ( do2DParam ) {
        if ( isPhiSymmetry ) {
                    h_orig_hitEnergy_alpha_r[i] = analyze.InitTH2(prefixall + "hist_hitenergy_alpha_radius", "", 8, 0, TMath::Pi(), nbinsy, ymin, ymax);
          hitspy_orig[i]->hist_hitenergy_alpha_absPhi_radius() = h_orig_hitEnergy_alpha_r[i];

        } else {
                    h_orig_hitEnergy_alpha_r[i] = analyze.InitTH2(prefixall + "hist_hitenergy_alpha_radius", "", 8, 0, 2 * TMath::Pi(), nbinsy, ymin, ymax);
          hitspy_orig[i]->hist_hitenergy_alpha_radius() = h_orig_hitEnergy_alpha_r[i];
        }
      }

      if ( doMeanRz ) {

        h_hitenergy_r[i] = analyze.InitTH1( prefixall + "hist_hitenergy_R", "1D", nbinsr, rmin, rmax );
        hitspy_orig[i]->hist_hitenergy_r() = h_hitenergy_r[i];
        h_orig_mean_hitenergy_r[i] = analyze.InitTH1( prefixall + "hist_mean_hitenergy_R", "1D", nbinsr, rmin, rmax );
        hitspy_orig[i]->hist_hitenergy_mean_r() = h_orig_mean_hitenergy_r[i];

        h_hitenergy_weight[i] = analyze.InitTH1( prefixall + "hist_hitenergy_weight", "1D", 2000, -2., 3. );
        hitspy_orig[i]->hist_hitenergy_weight() = h_hitenergy_weight[i];
	        h_orig_mean_hitenergy_weight[i] = analyze.InitTH1(prefixall + "hist_mean_hitenergy_weight", "1D", 2000, 0., 1.);
        hitspy_orig[i]->hist_hitenergy_mean_weight() = h_orig_mean_hitenergy_weight[i];

        h_hitenergy_z[i] = analyze.InitTH1( prefixall + "hist_hitenergy_z", "1D", nbinsz, zmin, zmax );
        hitspy_orig[i]->hist_hitenergy_z() = h_hitenergy_z[i];
        h_orig_mean_hitenergy_z[i] = analyze.InitTH1( prefixall + "hist_mean_hitenergy_z", "1D", nbinsz, zmin, zmax );
        hitspy_orig[i]->hist_hitenergy_mean_z() = h_orig_mean_hitenergy_z[i];

        h_Rz[i] = analyze.InitTH2( prefixall + "hist_Rz", "2D", 1000, 0, 5000, 1000, -10000, 10000 );
	        h_Rz_outOfRange[i] = analyze.InitTH2(prefixall + "hist_Rz_outOfRange", "2D", 1000, 0, 5000, 1000, -10000, 10000);
        hitspy_orig[i]->hist_Rz()            = h_Rz[i];
        hitspy_orig[i]->hist_Rz_outOfRange() = h_Rz_outOfRange[i];
      }


      if ( doZVertexStudies ) {
                h_deltaEtaAveragedPerEvent[i] = new TH1F((prefixall + "deltaEtaAveragedPerEvent").c_str(), "", 4000, -400, 400);
        h_deltaEta[i]       = new TH1F( ( prefixall + "deltaEtaHit" ).c_str(), "", 4000, -400, 400 );
        h_deltaPhi[i]       = new TH1F( ( prefixall + "deltaPhiHit" ).c_str(), "", 4000, -400, 400 );
        h_deltaRt[i]        = new TH1F( ( prefixall + "deltaRtHit" ).c_str(), "", 4000, -400, 400 );
        h_deltaZ[i]         = new TH1F( ( prefixall + "deltaZHit" ).c_str(), "", 4000, -400, 400 );
        h_energyPerLayer[i] = new TH1F( ( prefixall + "energyPerLayer" ).c_str(), "", 4000, 0, 65000 );
        h_MeanEnergy[i]     = new TH1F( ( prefixall + "meanEnergy" ).c_str(), "", 1, 0, 1 );

        h_deltaEtaAveragedPerEvent[i]->Sumw2();
        h_energyPerLayer[i]->Sumw2();
        input_EnergyAndHits[i]->hist_deltaEtaAveragedPerEvent() = h_deltaEtaAveragedPerEvent[i];
        input_EnergyAndHits[i]->hist_energyPerLayer()           = h_energyPerLayer[i];
        h_deltaEta[i]->Sumw2();
        h_deltaPhi[i]->Sumw2();
        h_deltaRt[i]->Sumw2();
        hitspy_orig[i]->hist_deltaEta() = h_deltaEta[i];
        hitspy_orig[i]->hist_deltaPhi() = h_deltaPhi[i];
        hitspy_orig[i]->hist_deltaRt()  = h_deltaRt[i];
        hitspy_orig[i]->hist_deltaZ()   = h_deltaZ[i];
      }

      input_EnergyAndHits[i]->push_back( centerPosCalc[i] );
      input_EnergyAndHits[i]->push_back( hitspy_orig[i] );
      analyze.validations().emplace_back( RunInputHits[i] );
    }

    std::cout << "=============================" << std::endl;
    //////////////////////////////////////////////////////////
    analyze.LoopEvents( -1 );

    for ( int ipca = 1; ipca <= npca; ipca++ ) {
      int i = ipca - 1;

      std::pair<double, double> energyMeanSigma = input_EnergyAndHits[i]->getMeanEnergyWithError();

      if ( doZVertexStudies ) {

        h_MeanEnergy[i]->SetBinContent( 1, energyMeanSigma.first );
        h_MeanEnergy[i]->SetBinError( 1, energyMeanSigma.second );
      }

      TH2F* h_alpha_r     = new TH2F();
      TH1F* h_zoom_mean_r = new TH1F();
      TH1F* h_zoom_mean_z = new TH1F();

      if ( do2DParam ) {
        h_alpha_r = (TH2F*)Create2DHistogram( h_orig_hitEnergy_alpha_r[i], energy_cutoff );
        CreatePolarPlot( h_alpha_r, plotDir.Data() ); // save polar plots
      }

      if ( doMeanRz ) {
        h_zoom_mean_r = zoomHisto( h_orig_mean_hitenergy_r[i] );
        h_zoom_mean_z = zoomHisto( h_orig_mean_hitenergy_z[i] );

        ( *tm_mean_weight )[analyze_layer][ipca] = h_orig_mean_hitenergy_weight[i]->GetMean();
      }

      if ( do2DParam ) {
        fshape->cd();
        h_alpha_r->Write();
      }

      if ( doMeanRz ) {
        fextrapol->cd();
        h_zoom_mean_r->Write();
        h_zoom_mean_z->Write();
        h_orig_mean_hitenergy_weight[i]->Write();
        h_Rz[i]->Write();
        h_Rz_outOfRange[i]->Write();
      }

      if ( doZVertexStudies ) {
        fzvertex->cd();
        h_deltaEtaAveragedPerEvent[i]->Write();
        h_deltaEta[i]->Write();
        h_deltaPhi[i]->Write();
        h_deltaRt[i]->Write();
        h_deltaZ[i]->Write();
        h_energyPerLayer[i]->Write();
        h_MeanEnergy[i]->Write();
      }

      if ( h_orig_hitEnergy_alpha_r[i] ) delete h_orig_hitEnergy_alpha_r[i];
      if ( h_alpha_r ) delete h_alpha_r;

      if ( h_deltaEtaAveragedPerEvent[i] ) delete h_deltaEtaAveragedPerEvent[i];
      if ( h_deltaEta[i] ) delete h_deltaEta[i];
      if ( h_deltaPhi[i] ) delete h_deltaPhi[i];
      if ( h_deltaRt[i] ) delete h_deltaRt[i];
      if ( h_deltaZ[i] ) delete h_deltaZ[i];
      if ( h_energyPerLayer[i] ) delete h_energyPerLayer[i];
      if ( h_MeanEnergy[i] ) delete h_MeanEnergy[i];

      if ( h_zoom_mean_r ) delete h_zoom_mean_r;
      if ( h_zoom_mean_z ) delete h_zoom_mean_z;
      if ( h_orig_mean_hitenergy_r[i] ) delete h_orig_mean_hitenergy_r[i];
      if ( h_orig_mean_hitenergy_z[i] ) delete h_orig_mean_hitenergy_z[i];
      if ( h_hitenergy_weight[i] ) delete h_hitenergy_weight[i];
      if ( h_orig_mean_hitenergy_weight[i] ) delete h_orig_mean_hitenergy_weight[i];
      if ( h_Rz[i] ) delete h_Rz[i];
      if ( h_Rz_outOfRange[i] ) delete h_Rz_outOfRange[i];


      if ( doZVertexStudies ) {
                std::cout << "\nDelta eta averaged in pca bin " << ipca << " over all events: " << input_EnergyAndHits[i]->getDeltaEtaAveraged() << std::endl;
                std::cout << "\n Energy in layer " << analyze_layer << " and pca " << ipca << " : " << energyMeanSigma.first << " +- " << energyMeanSigma.second <<  std::endl;
      }

      delete RunInputHits[i];
      delete input_EnergyAndHits[i];
      delete centerPosCalc[i];
      delete hitspy_orig[i];
    }
  }

  if ( doMeanRz ) {
    fextrapol->cd();
    tm_mean_weight->Write( "tm_mean_weight" );
  }

  if ( doMeanRz || useMeanRz ) {

    for ( size_t ilayer = 0; ilayer < v_layer.size(); ilayer++ ) {
            for (int ipca = 1; ipca <= npca; ipca++) {
                 std::cout << (*tm_mean_weight)[v_layer[ilayer]][ipca] << ",";
            }
      std::cout << std::endl;

    }
    delete tm_mean_weight;
  }
  if ( fshape ) fshape->Close();
  if ( fextrapol ) fextrapol->Close();
  if ( fzvertex ) fzvertex->Close();
}

static const char* USAGE =
    R"(Program to run shower shape parametrization

Usage:
  runTFCS2DParametrizationHistogram [options]

Options:
  -h --help                               Show help screen.
  --dsid <dsid>                           Sample dsid for shower shape parameterization. [default: 431004]
  --dsid_zv0 <dsid_zv0>                   Sample dsid used to make first PCA. For purposes of zvertex studies. If not set the same sample as for shower shape parameterization is used [default: -999].
  --sampleData <sampleData>               Path to sample list [default: InputSamplesList.txt].
  --topDir <topDir>                       Path to main output dir [default: ./output/].
  --version <version>                     Used to label output files [default: ver01].
  --energy_cutoff <energy_cutoff>         Energy cutoff to cut the shape histograms [default: 0.9995]. 
  --topPlotDir <topPlotDir>               Main directory to store plots. [default: output_plot].
  --do2DParam <do2DParam>                 Option to control 2D shower shape parameterization [default: 1].
  --isPhiSymmetry <isPhiSymetry>          Option to use phi symmetry in shower shape parameterization [default: 1].
  --doMeanRz <doMeanRZ>                   Option to calculate averaged shower center position [default: 0].
  --useMeanRz <useMeanRZ>                 Option to use averaged shower center position in shower shower shape parameterization [default: 0].
  --doZVertexStudies <doZVertexStudies>   Option to add additional output for ZVertex position studies to the output [default: 0].
  --seed <seed>                           Random seed [default: 42].
  --nEvents <nEvents>                     Number of events to run over with. All events will be used if nEvents<=0 [default: -1].
  --firstEvent <firstEvent>               Run will start from this event [default: 0].
  --debug <debug>                         Set debug level to print debug messages [default: 0].
  
)";



int main(int argc, char **argv)
{
  

  int         dsid             = 431004;
  int         dsid_zv0         = -999;
  std::string sampleData       = "InputSamplesList.txt";
  std::string topDir           = "./output/";
  std::string version          = "ver01";
  float       energy_cutoff    = 0.9995;
  std::string topPlotDir       = "output_plot/";
  bool        do2DParam        = true;
  bool        isPhiSymmetry    = true;
  bool        doMeanRz         = true;
  bool        useMeanRz        = false;
  bool        doZVertexStudies = false;
  long        seed             = 42;
  int         nEvents          = -1;
  int         firstEvent       = -1;
  int         debug            = -1;

  std::map<std::string, docopt::value> args = docopt::docopt( USAGE, {argv + 1, argv + argc}, true );

  try {

    try{ dsid=args["--dsid"].asLong();} catch(const std::invalid_argument& e)
      { throw std::invalid_argument("--dsid option is mandatory and it is expected to be an integer. Check input parameters."); }

    try{ dsid_zv0=args["--dsid_zv0"].asLong();} catch(const std::invalid_argument& e)
      { throw std::invalid_argument("--dsid_zv0 option is expected to be an integer. Check input parameters."); }

    try{ sampleData=args["--sampleData"].asString();} catch(const std::invalid_argument& e)
      { throw std::invalid_argument("--sampleData option is expected to be an integer. Check input parameters."); }

    try{ topDir=args["--topDir"].asString();} catch(const std::invalid_argument& e)
      { throw std::invalid_argument("--topDir option is expected to be an integer. Check input parameters."); }

    try{ version=args["--version"].asString();} catch(const std::invalid_argument& e)
      { throw std::invalid_argument("--version option is expected to be an integer. Check input parameters."); }

    try{ energy_cutoff=std::stof( args["--energy_cutoff"].asString() );} catch(const std::invalid_argument& e)
      { throw std::invalid_argument("--energy_cutoff option is expected to be a float. Check input parameters."); }

    try{ topPlotDir=args["--topPlotDir"].asString();} catch(const std::invalid_argument& e)
      { throw std::invalid_argument("--topPlotDir option is expected to be a string. Check input parameters."); }


    try{ do2DParam=args["--do2DParam"].asLong();} catch(const std::invalid_argument& e)
      { throw std::invalid_argument("--do2DParam option is expected to be an integer. Check input parameters."); }

    try{ isPhiSymmetry=args["--isPhiSymmetry"].asLong();} catch(const std::invalid_argument& e)
      { throw std::invalid_argument("--isPhiSymmetry option is expected to be an integer. Check input parameters."); }

    try{ doMeanRz=args["--doMeanRz"].asLong();} catch(const std::invalid_argument& e)
      { throw std::invalid_argument("--doMeanRz option is expected to be an integer. Check input parameters."); }

    try{ useMeanRz=args["--useMeanRz"].asLong();} catch(const std::invalid_argument& e)
      { throw std::invalid_argument("--useMeanRz option is expected to be an integer. Check input parameters."); }

    try{ doZVertexStudies=args["--doZVertexStudies"].asLong();} catch(const std::invalid_argument& e)
      { throw std::invalid_argument("--doZVertexStudies option is expected to be an integer. Check input parameters."); }

    try{ seed=args["--seed"].asLong();} catch(const std::invalid_argument& e)
      { throw std::invalid_argument("--seed option is expected to be an integer. Check input parameters."); }

    try{ nEvents=args["--nEvents"].asLong();} catch(const std::invalid_argument& e)
      { throw std::invalid_argument("--nEvents option is expected to be an integer. Check input parameters."); }
  
    try{ firstEvent=args["--firstEvent"].asLong();} catch(const std::invalid_argument& e)
      { throw std::invalid_argument("--firstEvent option is expected to be an integer. Check input parameters."); }
  
    try{ debug=args["--debug"].asLong();} catch(const std::invalid_argument& e)
      { throw std::invalid_argument("--debug option is expected to be an integer. Check input parameters."); }


  } catch ( const std::invalid_argument& e ) {
    std::cout << "Error: " << e.what() << std::endl << std::endl;
    std::cout << USAGE << std::endl;
    return -1;
  }

  if ( doMeanRz && useMeanRz ) {
    std::cout << "Error: doMeanRz and useMeanRz cannot be set as true at the same moment!" << std::endl;
    exit( -3 );
  }

  std::cout << std::endl
            << "Parameters are loaded successfully!"
            << std::endl
            << "Executing runTFCS2DParametrizationHistogram function."
            << std::endl;

  runTFCS2DParametrizationHistogram(dsid,dsid_zv0,sampleData,topDir,version,energy_cutoff,topPlotDir,do2DParam,isPhiSymmetry,doMeanRz,useMeanRz,doZVertexStudies,seed,nEvents,firstEvent,debug);

  return 0;
}
