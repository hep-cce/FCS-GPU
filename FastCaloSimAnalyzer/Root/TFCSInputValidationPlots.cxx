/*
  Copyright (C) 2002-2017 CERN for the benefit of the ATLAS collaboration
 */

#include "FastCaloSimAnalyzer/TFCSInputValidationPlots.h"

#include "atlasstyle/AtlasStyle.h"
#include "atlasstyle/AtlasLabels.h"
#include "atlasstyle/AtlasUtils.h"

#include "TROOT.h"
#include "TString.h"
#include "TMath.h"
#include "TFile.h"
#include "TH1.h"
#include "TH2.h"
#include "TChain.h"
#include "TCanvas.h"
#include "TLegend.h"
#include "TLatex.h"
#include "TStyle.h"
#include "TKey.h"
#include "TAxis.h"
#include "TPaletteAxis.h"

#include <iostream>
#include <iomanip>
#include <sstream>
#include <tuple>
#include <algorithm>
#include <fstream>

// #ifdef __CLING__
// // these are not headers - do not treat them as such - needed for ROOT6
// #include "atlasstyle/AtlasLabels.C"
// #include "atlasstyle/AtlasUtils.C"
// #endif

TFCSInputValidationPlots::TFCSInputValidationPlots() { m_debug = 0; }

TFCSInputValidationPlots::TFCSInputValidationPlots( TTree* tree, std::string outputfiledir, std::vector<int> vlayer ) {
  gROOT->SetBatch( kTRUE );

#ifdef __CINT__
  gROOT->LoadMacro( "atlasstyle/AtlasLabels.C" );
  gROOT->LoadMacro( "atlasstyle/AtlasUtils.C" );
#endif

  SetAtlasStyle();

  m_debug     = 0;
  m_tree      = tree;
  m_outputDir = outputfiledir;
  m_vlayer    = vlayer;

  TFCSAnalyzerBase::MakeColorVector();
}

TFCSInputValidationPlots::~TFCSInputValidationPlots() {}

void TFCSInputValidationPlots::PlotJOValidation( std::vector<std::string>* files, std::string var,
                                                 std::string xlabel ) {

  std::string name1 = "#pi^{+} + #pi^{-}";
  std::string name2 = "#pi^{#pm}";

  // std::cout << "file0 = " << (files->at(0)).c_str() << std::endl ;
  TFile* f1    = TFile::Open( ( files->at( 0 ) ).c_str() );
  TTree* tree1 = (TTree*)f1->Get( "FCS_flatNtuple" );
  // std::cout << "tree0 entry = " << tree1->GetEntries() << std::endl ;

  // std::cout << "file1 = " << (files->at(1)).c_str() << std::endl ;
  TFile* f2    = TFile::Open( ( files->at( 1 ) ).c_str() );
  TTree* tree2 = (TTree*)f2->Get( "FCS_flatNtuple" );
  // std::cout << "tree1 entry = " << tree2->GetEntries() << std::endl ;

  // std::cout << "file2 = " << (files->at(2)).c_str() << std::endl ;
  TFile* f3    = TFile::Open( ( files->at( 2 ) ).c_str() );
  TTree* tree3 = (TTree*)f3->Get( "FCS_flatNtuple" );
  // std::cout << "tree2 entry = " << tree3->GetEntries() << std::endl ;

  std::vector<int> v_layer = m_vlayer;
  double           xmin    = -3000;
  double           xmax    = 3000;

  std::string outDir = "JOValidation_mergebin/";

  system( ( "mkdir -p " + outDir ).c_str() );

  for ( unsigned int ilayer = 0; ilayer < v_layer.size(); ilayer++ ) {
    int layer = v_layer.at( ilayer );

    double merge = 5.;
    if ( layer == 1 or layer == 5 ) merge = 1.;
    int nbins = (int)( xmax - xmin ) / merge;

    std::string cut = "hit_energy*scale_factor*(layer==" + std::to_string( layer ) + ")";

    TH1F* h1   = new TH1F();
    TH1F* h2   = new TH1F();
    TH1F* h_pm = new TH1F();

    TFCSAnalyzerBase::GetTH1TTreeDraw( h1, tree1, var, &cut, nbins, xmin, xmax );
    TFCSAnalyzerBase::GetTH1TTreeDraw( h2, tree2, var, &cut, nbins, xmin, xmax );
    TFCSAnalyzerBase::GetTH1TTreeDraw( h_pm, tree3, var, &cut, nbins, xmin, xmax );

    TH1F* h_ppm = (TH1F*)h1->Clone();
    h_ppm->Add( h2 );
    h_pm->Scale( 1 / h_pm->Integral() );
    h_ppm->Scale( 1 / h_ppm->Integral() );

    TCanvas* c = TFCSAnalyzerBase::PlotTH1Ratio( h_ppm, h_pm,
                                                 "#pi, 65 GeV 0.20 < |#eta| < 0.25, layer " + std::to_string( layer ),
                                                 xlabel, name1, name2, "a.u.", name1 + "/" + name2 );

    std::string file = outDir + var + "_layer_" + std::to_string( layer );
    c->SaveAs( ( file + "_log" + ".png" ).c_str() );
    c->Close();

    delete h1;
    delete h2;
    delete h_pm;
    delete h_ppm;
    delete c;
  }
}

void TFCSInputValidationPlots::PlotTH1( std::string var, std::string xlabel ) {

  PlotTH1Layer( var, xlabel );
  PlotTH1PCA( var, xlabel );
}

void TFCSInputValidationPlots::PlotTH1Layer( std::string var, std::string xlabel ) {

  bool is_mm = false;
  if ( findWord( var, "mm" ) ) is_mm = true;

  double rmax = -1;
  if ( is_mm )
    rmax = GetMinRmax( "mm" );
  else
    rmax = GetMinRmax();

  binStruct bin;

  bin = GetBinValues( var, rmax );

  int nbins = bin.nbins;
  int xmin  = bin.min;
  int xmax  = bin.max;

  // std::cout << " nbins, xmin, xmax = " << nbins << " , " << xmin << " , " << xmax << std::endl;

  PlotTH1Layer( var, nbins, xmin, xmax, xlabel );
}

void TFCSInputValidationPlots::PlotTH1Layer( std::string var, int nbins, double xmin, double xmax,
                                             std::string xlabel ) {

  std::vector<int> v_layer = m_vlayer;
  TTree*           tree    = m_tree;

  std::string ylabel = "a.u.";

  TCanvas* c1     = new TCanvas( "c1", "", 800, 600 );
  TPad*    thePad = (TPad*)c1->cd();

  TLegend* leg = new TLegend( 0.7, 0.7, 0.95, 0.9 );
  leg->SetBorderSize( 0 );
  leg->SetFillStyle( 0 );
  leg->SetFillColor( 0 );
  leg->SetTextSize( 0.04 );

  for ( unsigned int ilayer = 0; ilayer < v_layer.size(); ilayer++ ) {
    int     layer = v_layer.at( ilayer );
    Color_t color = v_color.at( ilayer );

    double energy = GetEnergy( layer, 0 );

    TString histname( Form( "h%i", layer ) );
    // std::cout << " histname = " << histname << std::endl;
    TString legend( Form( "layer %i", layer ) );
    // std::cout << " label =" << label << std::endl;
    TString varexp( Form( "%s>>h%i(%i, %f, %f)", var.c_str(), layer, nbins, xmin, xmax ) );
    std::cout << " varexp = " << varexp << std::endl;
    TString cut( Form( "hit_energy*scale_factor*(layer==%i)", layer ) );
    std::cout << " cut =" << cut << std::endl;
    TString draw = "";
    if ( ilayer == 0 )
      draw = "hist EX2";
    else
      draw = "hist EX2 same";
    // std::cout << " draw = " << draw << std::endl;

    tree->Draw( varexp, cut, "goff" );
    TH1F* histo = (TH1F*)gROOT->FindObject( histname );
    histo->Sumw2();
    histo->Scale( 1 / energy );
    histo->GetXaxis()->SetTitle( xlabel.c_str() );
    histo->GetYaxis()->SetTitle( ylabel.c_str() );
    histo->SetLineColor( color );
    histo->SetMarkerColor( color );
    thePad->cd();
    if ( findWord( var, "mm" ) ) histo->SetAxisRange( xmin / 3, xmax / 3, "X" );
    histo->Draw( draw );
    gPad->Update();

    leg->AddEntry( histo, legend, "l" );
  }

  leg->Draw();
  ATLASLabel( 0.18, 0.05, "Simulation Internal" );
  TString label = TFCSAnalyzerBase::GetLabel();
  myText( 0.18, 0.9, 1, label );

  std::string outDir = m_outputDir;
  std::string ext    = ".pdf";
  std::string file   = outDir + var + "_compare_layer_pca0";

  c1->SaveAs( ( file + ".pdf" ).c_str() );
  c1->SaveAs( ( file + ".png" ).c_str() );

  c1->SetLogy();
  c1->SaveAs( ( file + "_log" + ".pdf" ).c_str() );
  c1->SaveAs( ( file + "_log" + ".png" ).c_str() );

  c1->Close();
}

void TFCSInputValidationPlots::PlotTH1PCA( std::string var, std::string xlabel ) {

  std::vector<int> v_layer = m_vlayer;

  bool is_mm = false;
  if ( findWord( var, "mm" ) ) is_mm = true;

  for ( unsigned int ilayer = 0; ilayer < v_layer.size(); ilayer++ ) {
    int layer = v_layer.at( ilayer );

    double rmax = -1;
    if ( is_mm )
      rmax = GetRmax( layer, 0, "mm" );
    else
      rmax = GetRmax( layer, 0 );

    binStruct bin;
    bin = GetBinValues( var, rmax );

    int nbins = bin.nbins;
    int xmin  = bin.min;
    int xmax  = bin.max;

    PlotTH1PCA( var, layer, nbins, xmin, xmax, xlabel );
  }
}

void TFCSInputValidationPlots::PlotTH1PCA( std::string var, int layer, int nbins, double xmin, double xmax,
                                           std::string xlabel ) {

  int npca = 6;

  TTree* tree = m_tree;

  std::string ylabel = "a.u.";

  TString  c( Form( "c%i", layer ) );
  TCanvas* c1     = new TCanvas( c, c, 800, 600 );
  TPad*    thePad = (TPad*)c1->cd();

  TLegend* leg = new TLegend( 0.7, 0.7, 0.95, 0.9 );
  leg->SetBorderSize( 0 );
  leg->SetFillStyle( 0 );
  leg->SetFillColor( 0 );
  leg->SetTextSize( 0.04 );

  for ( int ipca = 1; ipca < npca; ipca++ ) {

    int pca = ipca;

    Color_t color = v_color.at( pca );

    double energy = GetEnergy( layer, pca );

    TString histname( Form( "hl%ipca%i", layer, pca ) );
    // std::cout << " histname = " << histname << std::endl;

    TString legend( Form( "layer %i pca %i", layer, pca ) );
    // std::cout << " label =" << label << std::endl;

    TString varexp( Form( "%s>>hl%ipca%i(%i, %f, %f)", var.c_str(), layer, pca, nbins, xmin, xmax ) );
    std::cout << " varexp = " << varexp << std::endl;

    TString cut( Form( "hit_energy*scale_factor*(layer==%i && pca==%i)", layer, pca ) );
    std::cout << " cut =" << cut << std::endl;

    TString draw = "";
    if ( pca == 1 )
      draw = "hist EX2";
    else
      draw = "hist EX2 same";
    // std::cout << " draw = " << draw << std::endl;

    tree->Draw( varexp, cut, "goff" );
    TH1F* histo = (TH1F*)gROOT->FindObject( histname );
    histo->Sumw2();
    histo->Scale( 1 / energy );
    histo->GetXaxis()->SetTitle( xlabel.c_str() );
    histo->GetYaxis()->SetTitle( ylabel.c_str() );
    histo->SetLineColor( color );
    histo->SetMarkerColor( color );
    if ( findWord( var, "mm" ) ) histo->SetAxisRange( xmin / 3, xmax / 3, "X" );
    thePad->cd();
    histo->Draw( draw );
    gPad->Update();

    leg->AddEntry( histo, legend, "l" );

  } // end loop over pca

  leg->Draw();
  ATLASLabel( 0.18, 0.05, "Simulation Internal" );
  TString label = TFCSAnalyzerBase::GetLabel();
  myText( 0.18, 0.9, 1, label );

  std::string outDir = m_outputDir;
  std::string ext    = ".pdf";
  std::string file   = outDir + var + "_compare_pca_layer" + std::to_string( layer );

  c1->SaveAs( ( file + ".pdf" ).c_str() );
  c1->SaveAs( ( file + ".png" ).c_str() );
  c1->SetLogy();
  c1->SaveAs( ( file + "_log" + ".pdf" ).c_str() );
  c1->SaveAs( ( file + "_log" + ".png" ).c_str() );

  c1->Close();
}

void TFCSInputValidationPlots::PlotTH2( std::string var, std::string xlabel, std::string ylabel ) {

  int              npca    = 6;
  std::vector<int> v_layer = m_vlayer;

  bool is_mm = false;
  if ( findWord( var, "mm" ) ) is_mm = true;

  for ( unsigned int ilayer = 0; ilayer < v_layer.size(); ilayer++ ) {
    int layer = v_layer.at( ilayer );

    for ( int ipca = 0; ipca < npca; ipca++ ) {
      int pca = ipca;

      double rmax = -1;
      if ( is_mm )
        rmax = GetRmax( layer, pca, "mm" );
      else
        rmax = GetRmax( layer, pca );

      binStruct bin;
      bin = GetBinValues( var, rmax );

      int nbinsx = bin.nbins;
      int xmin   = bin.min;
      int xmax   = bin.max;

      int nbinsy = nbinsx;
      int ymin   = xmin;
      int ymax   = xmax;

      PlotTH2( var, layer, pca, nbinsx, xmin, xmax, nbinsy, ymin, ymax, xlabel, ylabel );
    }
  }
}

void TFCSInputValidationPlots::PlotTH2( std::string var, int layer, int pca, int nbinsx, double xmin, double xmax,
                                        int nbinsy, double ymin, double ymax, std::string xlabel, std::string ylabel ) {

  TTree* tree = m_tree;

  std::string zlabel = "";

  TString  c( Form( "c%i", layer ) );
  TCanvas* c1     = new TCanvas( c, c, 900, 600 );
  TPad*    thePad = (TPad*)c1->cd();
  thePad->SetGridx();
  thePad->SetGridy();
  thePad->SetLogz();
  thePad->SetLeftMargin( -3.5 );
  thePad->SetRightMargin( 3.5 );
  gStyle->SetPalette( kVisibleSpectrum );

  TString histname( Form( "hl%ipca%i", layer, pca ) );
  // std::cout << " histname = " << histname << std::endl;

  // std::cout << " label =" << label << std::endl;

  TString varexp( Form( "%s>>hl%ipca%i(%i, %f, %f, %i, %f, %f)", var.c_str(), layer, pca, nbinsx, xmin, xmax, nbinsy,
                        ymin, ymax ) );
  std::cout << " varexp = " << varexp << std::endl;

  TString cut = "";

  if ( pca == 0 ) {
    cut = Form( "hit_energy*scale_factor*(layer==%i)", layer );
  } else {
    cut = Form( "hit_energy*scale_factor*(layer==%i && pca==%i)", layer, pca );
  }

  std::cout << " cut =" << cut << std::endl;

  // cout << "Drawing histogram!" << endl;
  tree->Draw( varexp, cut, "goff" );
  // cout << "Histogram created using tree->Draw" << endl;
  // cout << "Searching for histogram using gROOT->FindObject" << endl;
  // std::cout << "2" << std::endl ;
  TH2F* histo = (TH2F*)gROOT->FindObject( histname );
  // cout << "Histogram found" << endl;
  histo->Sumw2();
  histo->GetXaxis()->SetTitle( xlabel.c_str() );
  histo->GetYaxis()->SetTitle( ylabel.c_str() );
  histo->GetYaxis()->SetTitleOffset( 0.75 );
  histo->GetXaxis()->SetLabelSize( 0.035 );
  histo->GetYaxis()->SetLabelSize( 0.035 );
  histo->GetZaxis()->SetTitle( zlabel.c_str() );
  histo->GetZaxis()->SetTitleOffset( 0.9 );
  histo->GetZaxis()->SetLabelSize( 0.035 );

  thePad->cd();
  histo->Draw( "colz" );

  gPad->Update();
  // cout << "Searching for palette" << endl;
  TPaletteAxis* palette = (TPaletteAxis*)histo->GetListOfFunctions()->FindObject( "palette" );
  // cout << "Palette found" << endl;
  // palette->GetAxis()->SetLabelSize(0.001);
  palette->SetX1NDC( 0.85 );
  palette->SetX2NDC( 0.9 );
  gPad->Modified();
  gPad->Update();

  std::string pca_label = "";
  if ( pca == 0 )
    pca_label = "pcaincl";
  else
    pca_label = "pca" + std::to_string( pca );

  ATLASLabel( 0.18, 0.05, "Simulation Internal" );
  TString label1 = TFCSAnalyzerBase::GetLabel();
  TString label  = Form( "%s, layer %i, %s", label1.Data(), layer, pca_label.c_str() );

  myText( 0.1, 0.96, 1, label );

  std::string var2 = TFCSAnalyzerBase::replaceChar( var, ':', '_' );
  // std::cout << "var2 = " << var2 << std::endl ;

  std::string outDir   = m_outputDir;
  std::string filename = var2 + "_layer" + std::to_string( layer ) + "_" + pca_label;
  std::string ext      = ".pdf";
  std::string file     = outDir + filename;
  // cout << "Saving histos into files!" << endl;
  c1->SaveAs( ( file + ".png" ).c_str() );
  c1->SaveAs( ( file + ".pdf" ).c_str() );

  // histo->Scale(1 / energy);
  // thePad->cd();
  // histo->Draw("colz");
  // gPad->Update();

  // c1->SaveAs((file + "_norm" + ext).c_str());

  // cout << "Closing TCanvas" << endl;
  c1->Close();
  delete histo;
  delete c1;
}

void TFCSInputValidationPlots::CreateBinning( double cutoff ) {

  std::cout << " * Creating automatic binning for global and mm coordinates " << std::endl;

  int    nbins   = 10000;
  double xmin    = 0.;
  double xmax    = 6.;
  double xmax_mm = 10000.;

  std::vector<int> v_layer = m_vlayer;

  TTree* tree = m_tree;

  for ( unsigned int ilayer = 0; ilayer < v_layer.size(); ilayer++ ) {
    int layer = v_layer.at( ilayer );

    for ( int ipca = 0; ipca < 6; ipca++ ) {
      std::cout << "CreateBinning: Running in layer " << layer << " , pca bin " << ipca << std::endl;
      int     pca = ipca;
      TString histname( Form( "hl%ipca%i", layer, pca ) );
      TString histname_mm( Form( "hlmm%ipca%i", layer, pca ) );

      TString varexp( Form( "%s>>hl%ipca%i(%i, %f, %f)", "radius", layer, pca, nbins, xmin, xmax ) );
      TString varexp_mm( Form( "%s>>hlmm%ipca%i(%i, %f, %f)", "radius_mm", layer, pca, nbins, xmin, xmax_mm ) );

      // std::cout << " varexp = " << varexp << std::endl;
      // std::cout << " varexp mm = " << varexp_mm << std::endl;

      TString cut = "";

      if ( pca == 0 ) {
        cut = Form( "TMath::Sqrt(2)*hit_energy*scale_factor*(layer==%i)", layer );
      } else {
        cut = Form( "TMath::Sqrt(2)*hit_energy*scale_factor*(layer==%i && pca==%i)", layer, pca );
      }

      // std::cout << " cut =" << cut << std::endl;
      TString draw = "goff";
      // std::cout << " draw = " << draw << std::endl;

      tree->Draw( varexp, cut, draw );
      TH1F* histo = (TH1F*)gROOT->FindObject( histname );

      double energy = histo->Integral();
      double rmax   = TFCSAnalyzerBase::GetBinUpEdge( histo, cutoff );
      // std::cout << " energy = " << energy << std::endl;
      // std::cout << "rmax = " << rmax << std::endl ;

      tree->Draw( varexp_mm, cut, draw );
      TH1F*  histo_mm = (TH1F*)gROOT->FindObject( histname_mm );
      double rmax_mm  = TFCSAnalyzerBase::GetBinUpEdge( histo_mm, cutoff );

      std::vector<double> v_Ermax;
      v_Ermax.push_back( energy );
      v_Ermax.push_back( rmax );
      v_Ermax.push_back( rmax_mm );

      EnergyRmax.insert( std::make_pair( std::make_pair( layer, pca ), v_Ermax ) );
    }
  }

  if ( m_debug ) {
    std::cout << "Retrieving Rmax " << std::endl;
    for ( unsigned int i = 0; i < v_layer.size(); i++ ) {
      for ( int j = 0; j < 6; j++ ) {
        std::vector<double> v_Ermax = GetEnergyRmax( v_layer.at( i ), j );
        std::cout << "energy = " << ( GetEnergyRmax( v_layer.at( i ), j ) ).at( 0 ) << std::endl;
        std::cout << " Rmax  = " << ( GetEnergyRmax( v_layer.at( i ), j ) ).at( 1 ) << std::endl;
        std::cout << " Rmax mm = " << ( GetEnergyRmax( v_layer.at( i ), j ) ).at( 2 ) << std::endl;
      }
    }
  }
}

std::vector<double> TFCSInputValidationPlots::GetEnergyRmax( int layer, int pca ) {

  return EnergyRmax[std::make_pair( layer, pca )];
}

double TFCSInputValidationPlots::GetRmax( int layer, int pca, std::string opt ) {
  double rmax;

  if ( opt == "mm" ) {
    rmax = ( EnergyRmax[std::make_pair( layer, pca )] ).at( 2 );
  } else
    rmax = ( EnergyRmax[std::make_pair( layer, pca )] ).at( 1 );

  return rmax;
}

double TFCSInputValidationPlots::GetEnergy( int layer, int pca ) {
  return ( EnergyRmax[std::make_pair( layer, pca )] ).at( 0 );
}

double TFCSInputValidationPlots::GetMaxRmax( std::string opt ) {
  std::vector<int> v_layer = m_vlayer;
  double           maxRmax = -1;

  for ( unsigned int ilayer = 0; ilayer < v_layer.size(); ilayer++ ) {
    int    layer = v_layer.at( ilayer );
    double rmax  = -1;
    if ( opt == "mm" )
      rmax = GetRmax( layer, 0, "mm" );
    else
      rmax = GetRmax( layer, 0 );

    if ( rmax > maxRmax ) maxRmax = rmax;
  }
  return maxRmax;
}

double TFCSInputValidationPlots::GetMinRmax( std::string opt ) {

  std::vector<int> v_layer = m_vlayer;
  double           minRmax = 1e9;

  for ( unsigned int ilayer = 0; ilayer < v_layer.size(); ilayer++ ) {
    int    layer = v_layer.at( ilayer );
    double rmax  = -1;
    if ( opt == "mm" )
      rmax = GetRmax( layer, 0, "mm" );
    else
      rmax = GetRmax( layer, 0 );
    if ( rmax < minRmax ) minRmax = rmax;
  }
  return minRmax;
}

TFCSInputValidationPlots::binStruct TFCSInputValidationPlots::GetBinValues( std::string var, double rmax ) {
  binStruct bin;

  if ( findWord( var, "mm" ) ) {
    if ( findWord( var, "radius_mm" ) ) {
      bin.nbins = (int)rmax + 1;
      bin.min   = 0;
      bin.max   = (int)rmax + 1;
    } else if ( findWord( var, "alpha_mm" ) ) {
      bin.nbins = 8;
      bin.min   = 0;
      bin.max   = 2.0 * TMath::Pi();
    } else {

      double value = (int)( rmax / TMath::Sqrt( 2 ) ) + 1;
      bin.nbins    = 2 * value;
      bin.min      = -value;
      bin.max      = value;
    }
  } else {
    float binwidth = 0.0006;
    if ( findWord( var, "radius" ) ) {
      bin.nbins = (int)( rmax / binwidth ) + 1;
      bin.min   = 0;
      bin.max   = rmax;
    } else if ( findWord( var, "alpha" ) ) {
      bin.nbins = 8;
      bin.min   = 0;
      bin.max   = 2.0 * TMath::Pi();
    } else {

      double value = rmax / TMath::Sqrt( 2.0 );
      bin.nbins    = (int)( ( 2 * value ) / binwidth ) + 1;
      bin.min      = -value;
      bin.max      = value;
    }
  }

  return bin;
}

void TFCSInputValidationPlots::CreateInputValidationHTML( std::string filename, std::vector<std::string> histNames ) {

  std::vector<int> v_layer = m_vlayer;
  std::string      outDir  = m_outputDir;

  std::string              ext = ".pdf";
  std::vector<std::string> plotNames;

  for ( unsigned int ihist = 0; ihist < histNames.size(); ihist++ ) {
    std::string hist = histNames.at( ihist );

    if ( findWord( hist, ":" ) ) {

      std::string name = TFCSAnalyzerBase::replaceChar( hist, ':', '_' );

      for ( unsigned int ilayer = 0; ilayer < v_layer.size(); ilayer++ ) {
        int layer = v_layer.at( ilayer );
        for ( int ipca = 0; ipca < 6; ipca++ ) {
          int         pca       = ipca;
          std::string pca_label = "";
          if ( pca == 0 )
            pca_label = "_pcaincl";
          else
            pca_label = "_pca" + std::to_string( pca );
          plotNames.push_back( name + "_layer" + std::to_string( layer ) + pca_label );
        }
      }

    } else {

      plotNames.push_back( hist + "_compare_layer_pca0_log" );

      for ( unsigned int ilayer = 0; ilayer < v_layer.size(); ilayer++ ) {
        int layer = v_layer.at( ilayer );

        plotNames.push_back( hist + "_compare_pca_layer" + std::to_string( layer ) + "_log" );
      }
    }
  }

  TFCSAnalyzerBase::CreateHTML( filename, plotNames );
  std::cout << "Created HTML at: " << filename.c_str() << std::endl;
}
