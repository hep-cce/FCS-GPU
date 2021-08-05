/*
  Copyright (C) 2002-2017 CERN for the benefit of the ATLAS collaboration
*/

#include "FastCaloSimAnalyzer/TFCSAnalyzerBase.h"

#include "TROOT.h"
#include "TString.h"
#include "TMath.h"
#include "TFile.h"
#include "TH1.h"
#include "TH2.h"
#include "TH1D.h"
#include "TProfile.h"
#include "TProfile2D.h"
#include "TString.h"
#include "TCanvas.h"
#include "TLegend.h"
#include "TLatex.h"
#include "TStyle.h"
#include "TKey.h"
#include "TAxis.h"
#include "TPaletteAxis.h"

#include <iostream>
#include <tuple>
#include <fstream>

using namespace std;

TFCSAnalyzerBase::TFCSAnalyzerBase() {}

TFCSAnalyzerBase::~TFCSAnalyzerBase() {

  if ( m_cellVector ) delete m_cellVector;
  if ( m_avgcellVector ) delete m_avgcellVector;
  if ( m_truthCollection ) delete m_truthCollection;
  if ( m_TTC_entrance_OK ) delete m_TTC_entrance_OK;
  if ( m_TTC_entrance_eta ) delete m_TTC_entrance_eta;
  if ( m_TTC_entrance_phi ) delete m_TTC_entrance_phi;
  if ( m_TTC_entrance_r ) delete m_TTC_entrance_r;
  if ( m_TTC_entrance_z ) delete m_TTC_entrance_z;
  if ( m_TTC_mid_OK ) delete m_TTC_mid_OK;
  if ( m_TTC_mid_eta ) delete m_TTC_mid_eta;
  if ( m_TTC_mid_phi ) delete m_TTC_mid_phi;
  if ( m_TTC_mid_r ) delete m_TTC_mid_r;
  if ( m_TTC_mid_z ) delete m_TTC_mid_z;
  if ( m_TTC_back_OK ) delete m_TTC_back_OK;
  if ( m_TTC_back_eta ) delete m_TTC_back_eta;
  if ( m_TTC_back_phi ) delete m_TTC_back_phi;
  if ( m_TTC_back_r ) delete m_TTC_back_r;
  if ( m_TTC_back_z ) delete m_TTC_back_z;
  if ( m_TTC_IDCaloBoundary_eta ) delete m_TTC_IDCaloBoundary_eta;
  if ( m_TTC_IDCaloBoundary_phi ) delete m_TTC_IDCaloBoundary_phi;
  if ( m_TTC_IDCaloBoundary_r ) delete m_TTC_IDCaloBoundary_r;
  if ( m_TTC_IDCaloBoundary_z ) delete m_TTC_IDCaloBoundary_z;
}

TH1F* TFCSAnalyzerBase::InitTH1( std::string histname, std::string histtype, int nbins, float low, float high,
                                 std::string xtitle, std::string ytitle ) {

  string fullname = histname + "_" + histtype;

  TH1F* hist = new TH1F( fullname.c_str(), fullname.c_str(), nbins, low, high );
  hist->GetXaxis()->SetTitle( xtitle.c_str() );
  hist->GetYaxis()->SetTitle( ytitle.c_str() );

  if ( std::find( histNameVec.begin(), histNameVec.end(), histname ) == histNameVec.end() ) {
    histNameVec.push_back( fullname );
    histVec.push_back( hist );
  }
  m_histMap[fullname] = hist;

  return hist;
}

TH2F* TFCSAnalyzerBase::InitTH2( std::string histname, std::string histtype, int nbinsx, float lowx, float highx,
                                 int nbinsy, float lowy, float highy, std::string xtitle, std::string ytitle ) {

  string fullname = histname + "_" + histtype;

  TH2F* hist = new TH2F( fullname.c_str(), fullname.c_str(), nbinsx, lowx, highx, nbinsy, lowy, highy );
  hist->GetXaxis()->SetTitle( xtitle.c_str() );
  hist->GetYaxis()->SetTitle( ytitle.c_str() );

  if ( std::find( histNameVec.begin(), histNameVec.end(), histname ) == histNameVec.end() ) {
    histNameVec.push_back( fullname );
    histVec.push_back( hist );
  }
  m_histMap[fullname] = hist;

  return hist;
}

TProfile* TFCSAnalyzerBase::InitTProfile1D( std::string histname, std::string histtype, int nbinsx, float lowx,
                                            float highx, std::string xtitle, std::string ytitle,
                                            std::string profiletype ) {

  string fullname = histname + "_" + histtype;

  TProfile* hist = new TProfile( fullname.c_str(), fullname.c_str(), nbinsx, lowx, highx, profiletype.c_str() );
  hist->GetXaxis()->SetTitle( xtitle.c_str() );
  hist->GetYaxis()->SetTitle( ytitle.c_str() );

  if ( std::find( histNameVec.begin(), histNameVec.end(), histname ) == histNameVec.end() ) {
    histNameVec.push_back( fullname );
    histVec.push_back( hist );
  }
  m_histMap[fullname] = hist;

  return hist;
}

TProfile2D* TFCSAnalyzerBase::InitTProfile2D( std::string histname, std::string histtype, int nbinsx, float lowx,
                                              float highx, int nbinsy, float lowy, float highy, std::string xtitle,
                                              std::string ytitle, std::string profiletype ) {

  string fullname = histname + "_" + histtype;

  TProfile2D* hist = new TProfile2D( fullname.c_str(), fullname.c_str(), nbinsx, lowx, highx, nbinsy, lowy, highy,
                                     profiletype.c_str() );
  hist->GetXaxis()->SetTitle( xtitle.c_str() );
  hist->GetYaxis()->SetTitle( ytitle.c_str() );

  if ( std::find( histNameVec.begin(), histNameVec.end(), histname ) == histNameVec.end() ) {
    histNameVec.push_back( fullname );
    histVec.push_back( hist );
  }
  m_histMap[fullname] = hist;

  return hist;
}

void TFCSAnalyzerBase::Fill( TH1* h, float value, float weight ) {
  TAxis* x           = h->GetXaxis();
  int    nbins       = x->GetNbins();
  float  max         = x->GetBinUpEdge( nbins );
  float  min         = x->GetBinLowEdge( 1 );
  float  thisvalue   = value;
  float  width_first = h->GetBinWidth( 1 );
  float  width_last  = h->GetBinWidth( nbins );
  if ( thisvalue > max ) thisvalue = max - width_last / 2;
  if ( thisvalue < min ) thisvalue = min + width_first / 2;
  h->Fill( thisvalue, weight );
}

void TFCSAnalyzerBase::Fill( TH2* h, float valuex, float valuey, float weight ) {
  TAxis* x            = h->GetXaxis();
  int    nbinsx       = x->GetNbins();
  float  xmax         = x->GetBinUpEdge( nbinsx );
  float  xmin         = x->GetBinLowEdge( 1 );
  float  thisvaluex   = valuex;
  float  widthx_first = x->GetBinWidth( 1 );
  float  widthx_last  = x->GetBinWidth( nbinsx );
  if ( thisvaluex > xmax ) thisvaluex = xmax - widthx_last / 2;
  if ( thisvaluex < xmin ) thisvaluex = xmin + widthx_first / 2;

  TAxis* y            = h->GetYaxis();
  int    nbinsy       = y->GetNbins();
  float  ymax         = y->GetBinUpEdge( nbinsy );
  float  ymin         = y->GetBinLowEdge( 1 );
  float  thisvaluey   = valuey;
  float  widthy_first = y->GetBinWidth( 1 );
  float  widthy_last  = y->GetBinWidth( nbinsy );
  if ( thisvaluey > ymax ) thisvaluey = ymax - widthy_last / 2;
  if ( thisvaluey < ymin ) thisvaluey = ymin + widthy_first / 2;

  h->Fill( thisvaluex, thisvaluey, weight );
}

void TFCSAnalyzerBase::Fill( TProfile* h, float valuex, float valuey, float weight ) {
  TAxis* x            = h->GetXaxis();
  int    nbinsx       = x->GetNbins();
  float  xmax         = x->GetBinUpEdge( nbinsx );
  float  xmin         = x->GetBinLowEdge( 1 );
  float  thisvaluex   = valuex;
  float  widthx_first = x->GetBinWidth( 1 );
  float  widthx_last  = x->GetBinWidth( nbinsx );
  if ( thisvaluex > xmax ) thisvaluex = xmax - widthx_last / 2;
  if ( thisvaluex < xmin ) thisvaluex = xmin + widthx_first / 2;

  h->Fill( thisvaluex, valuey, weight );
}

void TFCSAnalyzerBase::Fill( TProfile2D* h, float valuex, float valuey, float valuez, float weight ) {
  TAxis* x            = h->GetXaxis();
  int    nbinsx       = x->GetNbins();
  float  xmax         = x->GetBinUpEdge( nbinsx );
  float  xmin         = x->GetBinLowEdge( 1 );
  float  thisvaluex   = valuex;
  float  widthx_first = x->GetBinWidth( 1 );
  float  widthx_last  = x->GetBinWidth( nbinsx );
  if ( thisvaluex > xmax ) thisvaluex = xmax - widthx_last / 2;
  if ( thisvaluex < xmin ) thisvaluex = xmin + widthx_first / 2;

  TAxis* y            = h->GetYaxis();
  int    nbinsy       = y->GetNbins();
  float  ymax         = y->GetBinUpEdge( nbinsy );
  float  ymin         = y->GetBinLowEdge( 1 );
  float  thisvaluey   = valuey;
  float  widthy_first = y->GetBinWidth( 1 );
  float  widthy_last  = y->GetBinWidth( nbinsy );
  if ( thisvaluey > ymax ) thisvaluey = ymax - widthy_last / 2;
  if ( thisvaluey < ymin ) thisvaluey = ymin + widthy_first / 2;

  h->Fill( thisvaluex, thisvaluey, valuez, weight );
}

void TFCSAnalyzerBase::autozoom( TH1* h1, double& min, double& max, double& rmin, double& rmax ) {

  double min1, min2, max1, max2;
  min1 = min2 = h1->GetXaxis()->GetXmin();
  max1 = max2 = h1->GetXaxis()->GetXmax();

  for ( int b = 1; b <= h1->GetNbinsX(); b++ ) {
    if ( h1->GetBinContent( b ) > 0 ) {
      min1 = h1->GetBinCenter( b );
      break;
    }
  }
  for ( int b = h1->GetNbinsX(); b >= 1; b-- ) {
    if ( h1->GetBinContent( b ) > 0 ) {
      max1 = h1->GetBinCenter( b );
      break;
    }
  }

  min = min1;
  max = max1;

  rmin = min - 0.5 * h1->GetBinWidth( 1 );
  rmax = max + 0.5 * h1->GetBinWidth( 1 );
}

TH1D* TFCSAnalyzerBase::refill( TH1* h_in, double min, double max, double rmin, double rmax ) {

  // int debug=0;

  int Nbins;
  int bins = 0;
  for ( int b = h_in->FindBin( min ); b <= h_in->FindBin( max ); b++ ) bins++;

  if ( bins <= 120 ) {
    // no rebinning
    Nbins = bins;
  } else {
    int tries = 0;
    int rebin = 2;

    while ( tries < 1000 ) {
      if ( ( 10000 % rebin ) == 0 ) {
        TH1D* h_clone = (TH1D*)h_in->Clone( "h_clone" );
        h_clone->Rebin( rebin );
        Nbins = 0;
        for ( int b = h_clone->FindBin( min ); b <= h_clone->FindBin( max ); b++ ) Nbins++;
        if ( Nbins < 120 && Nbins > 50 ) {
          h_in->Rebin( rebin );
          //          cout << "*decide for rebin=" << rebin << "*" << endl;
          break;
        }
      }
      rebin++;
      tries++;
    }
    if ( tries >= 1000 ) {
      cout << " ********** GIVE UP ********** " << endl;
      h_in->Rebin( (double)bins / 100.0 );
      Nbins = 0;
      for ( int b = h_in->FindBin( min ); b <= h_in->FindBin( max ); b++ ) Nbins++;
    }
  }

  // if(debug) cout<<"---> NBINS "<<Nbins<<endl;

  int start = h_in->FindBin( min ) - 1;

  // if(debug) cout<<"AFTER rebin ->underflow "<<h_in->GetBinContent(0)<<" startbin "<<start<<" minimum "<<min<<endl;

  TH1D* h_out = new TH1D( TString( "rebin_" ) + h_in->GetName(), h_in->GetTitle(), Nbins, rmin, rmax );
  h_out->SetXTitle( h_in->GetXaxis()->GetTitle() );
  h_out->SetYTitle( h_in->GetYaxis()->GetTitle() );
  for ( int b = 1; b <= h_out->GetNbinsX(); b++ ) {
    h_out->SetBinContent( b, h_in->GetBinContent( start + b ) );
    h_out->SetBinError( b, h_in->GetBinError( start + b ) );
  }

  // if(debug) cout<<"AFTER refill ->underflow "<<h_out->GetBinContent(0)<<" startbin "<<start<<" minimum "<<min<<endl;

  return h_out;
}

void TFCSAnalyzerBase::GetTH1TTreeDraw( TH1F*& hist, TTree* tree, std::string var, std::string* cut, int nbins,
                                        double xmin, double xmax ) {

  TH1F*   histo = new TH1F();
  TString varexp( Form( "%s>>histo(%i, %f, %f)", var.c_str(), nbins, xmin, xmax ) );

  // std::cout << "varexp = " << varexp << std::endl ;
  TString selection( Form( "%s", cut->c_str() ) );
  // std::cout << "selection = " << selection << std::endl ;
  tree->Draw( varexp, selection, "goff" );

  TH1F* htemp = (TH1F*)gROOT->FindObject( "histo" );
  hist        = (TH1F*)htemp->Clone( "hist" );

  delete htemp;
  delete histo;
}

void TFCSAnalyzerBase::GetTH2TTreeDraw( TH2F*& hist, TTree* tree, std::string var, std::string* cut, int nbinsx,
                                        double xmin, double xmax, int nbinsy, double ymin, double ymax ) {

  TH2F*   histo = new TH2F();
  TString varexp( Form( "%s>>histo(%i, %f, %f, %i, %f, %f)", var.c_str(), nbinsx, xmin, xmax, nbinsy, ymin, ymax ) );

  // std::cout << "varexp = " << varexp << std::endl ;
  TString selection( Form( "%s", cut->c_str() ) );
  // std::cout << "selection = " << selection << std::endl ;
  tree->Draw( varexp, selection, "goff" );

  TH2F* htemp = (TH2F*)gROOT->FindObject( "histo" );
  hist        = (TH2F*)htemp->Clone( "hist" );

  delete htemp;
  delete histo;
}

TCanvas* TFCSAnalyzerBase::PlotPolar( TH2F* h, std::string label, std::string xlabel, std::string ylabel,
                                      std::string zlabel, int zoom_level ) {

  gStyle->SetPalette( kRainBow );
  gStyle->SetOptStat( 0 );

  h->Sumw2();

  int   nzoom = h->GetNbinsY() / zoom_level;
  float zoom  = h->GetYaxis()->GetBinUpEdge( nzoom );

  h->GetYaxis()->SetRangeUser( -float( zoom ), float( zoom ) );
  h->GetYaxis()->SetLabelSize( .025 );
  h->GetXaxis()->SetLabelSize( .025 );
  h->GetXaxis()->SetTitle( xlabel.c_str() );
  h->GetXaxis()->SetTitleSize( 0.035 );
  h->GetYaxis()->SetTitle( ylabel.c_str() );
  h->GetYaxis()->SetTitleSize( 0.035 );

  h->GetZaxis()->SetLabelSize( 0.025 );
  h->GetZaxis()->SetTitle( zlabel.c_str() );
  h->GetZaxis()->SetTitleSize( 0.035 );
  h->GetZaxis()->SetTitleOffset( 1.4 );

  TLatex* title = new TLatex( -zoom, 1.02 * zoom, label.c_str() );
  title->SetTextSize( 0.03 );
  title->SetTextFont( 42 );

  TLatex* l = new TLatex( -1 * zoom, -1.20 * zoom, "ATLAS" );
  l->SetTextSize( .035 );
  l->SetTextFont( 72 );

  TLatex* l2 = new TLatex( -0.6 * zoom, -1.20 * zoom, "Simulation Internal" );
  // TLatex* l2 = new TLatex(-0.6 * zoom, -1.20 * zoom, "Simulation Preliminary");

  l2->SetTextSize( .035 );
  l2->SetTextFont( 42 );

  TCanvas* c1 = new TCanvas( "c1", "", 900, 800 );
  c1->cd();
  c1->SetLeftMargin( 0.14 );
  c1->SetRightMargin( 0.17 );

  std::string frameTitle = "; " + xlabel + "; " + ylabel;
  gPad->DrawFrame( -zoom, -zoom, zoom, zoom, frameTitle.c_str() );
  h->Draw( "same colz pol" );
  l->Draw();
  l2->Draw();
  title->Draw();
  c1->SetLogz();

  return c1;
}

TCanvas* TFCSAnalyzerBase::PlotTH1Ratio( TH1F* h1, TH1F* h2, std::string label, std::string xlabel, std::string leg1,
                                         std::string leg2, std::string ylabel1, std::string ylabel2 ) {

  // std::cout << "Ratio plots for two samples ...." << std::endl ;

  TCanvas* c1   = new TCanvas( "c1", "", 660, 720 );
  TPad*    pad1 = new TPad( "pad1", "pad1", 0, 0.3525, 1, 1 );

  TPad* pad2 = new TPad( "pad2", "pad2", 0, 0, 1, 0.35 - 0.0025 );

  pad2->SetTopMargin( 0.06 );
  pad1->SetBottomMargin( 0.001 );
  pad2->SetBottomMargin( 0.3 );
  pad1->SetLeftMargin( 0.13 );
  pad2->SetLeftMargin( 0.13 );
  pad1->SetRightMargin( 0.075 );
  pad2->SetRightMargin( 0.075 );

  TLatex* title = new TLatex( 0.18, 0.9, label.c_str() );
  title->SetTextSize( 0.03 );
  title->SetTextFont( 42 );

  TLatex* l = new TLatex( 0.18, 0.05, "ATLAS" );
  l->SetTextSize( .035 );
  l->SetTextFont( 72 );

  TLatex* l2 = new TLatex( 0.2, 0.05, "Simulation Internal" );
  l2->SetTextSize( .035 );
  l2->SetTextFont( 42 );

  c1->cd();
  pad1->Draw();
  pad2->Draw();

  pad1->SetLogy();

  TLegend* leg = new TLegend( 0.7, 0.7, 0.95, 0.9 );
  leg->SetBorderSize( 0 );
  leg->SetFillStyle( 0 );
  leg->SetFillColor( 0 );
  leg->SetTextSize( 0.04 );

  h1->Sumw2();
  h1->GetXaxis()->SetTitle( xlabel.c_str() );
  h1->GetYaxis()->SetTitle( ylabel1.c_str() );
  h1->SetLineColor( kRed + 2 );
  h1->SetMarkerColor( kRed + 2 );

  h2->Sumw2();
  h2->GetXaxis()->SetTitle( xlabel.c_str() );
  h2->GetYaxis()->SetTitle( ylabel1.c_str() );
  h2->SetLineColor( kBlue + 2 );
  h2->SetMarkerColor( kBlue + 2 );

  pad1->cd();
  h1->Draw();
  h2->Draw( "same" );

  leg->AddEntry( h1, leg1.c_str(), "l" );
  leg->AddEntry( h2, leg2.c_str(), "l" );
  leg->Draw( "same" );

  l->Draw();
  l2->Draw();
  title->Draw();

  // ATLASLabel(0.18, 0.05, "Simulation Internal");
  // myText(0.18, 0.9, 1, label.c_str());

  pad2->cd();

  TH1F* hratio = (TH1F*)h1->Clone( "hratio" );
  hratio->Divide( h2 );
  hratio->SetLineColor( kGray );
  hratio->SetMarkerColor( kBlack );
  hratio->GetXaxis()->SetTitle( xlabel.c_str() );
  hratio->GetYaxis()->SetTitle( ylabel2.c_str() );
  hratio->GetXaxis()->SetLabelSize( .09 );
  hratio->GetYaxis()->SetLabelSize( .09 );
  hratio->GetXaxis()->SetTitleSize( .09 );
  hratio->GetYaxis()->SetTitleSize( .09 );
  hratio->GetYaxis()->SetTitleOffset( 0.7 );

  hratio->GetYaxis()->SetRangeUser( -0.5, 2.5 );
  hratio->Draw( "p" );

  TLine* line = new TLine( hratio->GetXaxis()->GetBinLowEdge( 1 ), 1.0,
                           hratio->GetXaxis()->GetBinUpEdge( hratio->GetNbinsX() ), 1.0 );
  line->SetLineColor( kRed );
  line->Draw();
  gPad->RedrawAxis();
  gPad->Update();

  return c1;
}

std::tuple<float, float> TFCSAnalyzerBase::GetUnitsmm( float eta_hit, float d_eta, float d_phi, CaloCell* cell ) {
  float phi_dist2r = 1.0;
  float cell_r     = cell->r;
  float cell_z     = cell->z;

  float dist000 = TMath::Sqrt( cell_r * cell_r + cell_z * cell_z );

  float eta_jakobi = TMath::Abs( 2.0 * TMath::Exp( -eta_hit ) / ( 1.0 + TMath::Exp( -2 * eta_hit ) ) );

  d_eta = d_eta * eta_jakobi * dist000;
  d_phi = d_phi * cell_r * phi_dist2r;

  return std::make_tuple( d_eta, d_phi );
}

std::tuple<float, float> TFCSAnalyzerBase::GetUnitsmm( float eta_hit, float d_eta, float d_phi, float cell_r,
                                                       float cell_z ) {
  float phi_dist2r = 1.0;

  float dist000 = TMath::Sqrt( cell_r * cell_r + cell_z * cell_z );

  float eta_jakobi = TMath::Abs( 2.0 * TMath::Exp( -eta_hit ) / ( 1.0 + TMath::Exp( -2 * eta_hit ) ) );

  d_eta = d_eta * eta_jakobi * dist000;
  d_phi = d_phi * cell_r * phi_dist2r;

  return std::make_tuple( d_eta, d_phi );
}

double TFCSAnalyzerBase::GetParticleMass( int pdgid ) {

  // * particle masses (MeV)
  std::map<int, double> pid_mass = {{11, 0.5}, {22, 0}, {211, 139.6}, {2212, 938.2}, {2112, 939.6}, {321, 493.7}};

  std::map<int, double>::iterator it;

  int apid = TMath::Abs( pdgid );
  it       = pid_mass.find( apid );

  if ( it != pid_mass.end() ) {
    return it->second;
  } else {
    std::cerr << "ERROR: Cannot find the mass of the particle" << std::endl;
  }
  return 0;
}

double TFCSAnalyzerBase::Mom2Etot( double mass, double mom ) {
  double Etot = TMath::Sqrt( mom * mom + mass * mass );
  return Etot;
}

double TFCSAnalyzerBase::Mom2Etot( int pdgid, double mom ) {
  double mass = GetParticleMass( pdgid );
  double Etot = Mom2Etot( mass, mom );
  return Etot;
}

double TFCSAnalyzerBase::Mom2Ekin( int pdgid, double mom ) {
  double mass = GetParticleMass( pdgid );
  double Ekin = Mom2Etot( mass, mom ) - mass;

  return Ekin;
}

double TFCSAnalyzerBase::Mom2Ekin_min( int pdgid, double mom ) {

  double mom_min = mom / TMath::Sqrt( 2 );
  return Mom2Ekin( pdgid, mom_min );
}

double TFCSAnalyzerBase::Mom2Ekin_max( int pdgid, double mom ) {
  double mom_max = TMath::Sqrt( 2 ) * mom;
  return Mom2Ekin( pdgid, mom_max );
}

void TFCSAnalyzerBase::InitInputTree( TChain* mychain, int /*layer*/ ) {

  m_branches.clear();

  m_cellVector      = nullptr;
  m_avgcellVector   = nullptr;
  m_truthCollection = nullptr;
  m_truthPx         = nullptr;
  m_truthPy         = nullptr;
  m_truthPz         = nullptr;
  m_truthE          = nullptr;
  m_truthPDGID      = nullptr;

  m_TTC_entrance_OK  = nullptr;
  m_TTC_entrance_eta = nullptr;
  m_TTC_entrance_phi = nullptr;
  m_TTC_entrance_r   = nullptr;
  m_TTC_entrance_z   = nullptr;

  m_TTC_mid_OK  = nullptr;
  m_TTC_mid_eta = nullptr;
  m_TTC_mid_phi = nullptr;
  m_TTC_mid_r   = nullptr;
  m_TTC_mid_z   = nullptr;

  m_TTC_back_OK  = nullptr;
  m_TTC_back_eta = nullptr;
  m_TTC_back_phi = nullptr;
  m_TTC_back_r   = nullptr;
  m_TTC_back_z   = nullptr;

  m_TTC_IDCaloBoundary_eta = nullptr;
  m_TTC_IDCaloBoundary_phi = nullptr;
  m_TTC_IDCaloBoundary_r   = nullptr;
  m_TTC_IDCaloBoundary_z   = nullptr;

  m_total_hit_energy  = 0.;
  m_total_cell_energy = 0.;
  m_pca               = 0;

  m_total_layer_cell_energy.resize( 30 );
  for ( int i = 0; i < 30; ++i ) m_total_layer_cell_energy[i] = 0;
  m_total_energy = 0;

  // TString b_Sampling = Form("Sampling_%i", layer);
  // TString b_AvgSampling = Form("AvgSampling_%i", layer);

  // TODO: check where these are used
  // setBranch(mychain, m_branches, b_Sampling, &m_cellVector);
  // setBranch(mychain, m_branches, b_AvgSampling, &m_avgcellVector);

  // setBranch(mychain, m_branches, "TruthCollection", &m_truthCollection);
  setBranch( mychain, m_branches, "TruthPx", &m_truthPx );
  setBranch( mychain, m_branches, "TruthPy", &m_truthPy );
  setBranch( mychain, m_branches, "TruthPz", &m_truthPz );
  setBranch( mychain, m_branches, "TruthE", &m_truthE );
  setBranch( mychain, m_branches, "TruthPDG", &m_truthPDGID );

  setBranch( mychain, m_branches, "newTTC_entrance_OK", &m_TTC_entrance_OK );
  setBranch( mychain, m_branches, "newTTC_entrance_eta", &m_TTC_entrance_eta );
  setBranch( mychain, m_branches, "newTTC_entrance_phi", &m_TTC_entrance_phi );
  setBranch( mychain, m_branches, "newTTC_entrance_r", &m_TTC_entrance_r );
  setBranch( mychain, m_branches, "newTTC_entrance_z", &m_TTC_entrance_z );

  setBranch( mychain, m_branches, "newTTC_mid_OK", &m_TTC_mid_OK );
  setBranch( mychain, m_branches, "newTTC_mid_eta", &m_TTC_mid_eta );
  setBranch( mychain, m_branches, "newTTC_mid_phi", &m_TTC_mid_phi );
  setBranch( mychain, m_branches, "newTTC_mid_r", &m_TTC_mid_r );
  setBranch( mychain, m_branches, "newTTC_mid_z", &m_TTC_mid_z );

  setBranch( mychain, m_branches, "newTTC_back_OK", &m_TTC_back_OK );
  setBranch( mychain, m_branches, "newTTC_back_eta", &m_TTC_back_eta );
  setBranch( mychain, m_branches, "newTTC_back_phi", &m_TTC_back_phi );
  setBranch( mychain, m_branches, "newTTC_back_r", &m_TTC_back_r );
  setBranch( mychain, m_branches, "newTTC_back_z", &m_TTC_back_z );

  setBranch( mychain, m_branches, "newTTC_IDCaloBoundary_eta", &m_TTC_IDCaloBoundary_eta );
  setBranch( mychain, m_branches, "newTTC_IDCaloBoundary_phi", &m_TTC_IDCaloBoundary_phi );
  setBranch( mychain, m_branches, "newTTC_IDCaloBoundary_r", &m_TTC_IDCaloBoundary_r );
  setBranch( mychain, m_branches, "newTTC_IDCaloBoundary_z", &m_TTC_IDCaloBoundary_z );

  // setBranch(mychain, m_branches, "total_hit_energy", &m_total_hit_energy);
  // setBranch(mychain, m_branches, "total_cell_energy", &m_total_cell_energy);

  // setBranch(mychain, m_branches, "firstPCAbin", &m_pca);
  // setBranch(mychain, m_branches, "energy_totalE", &m_total_energy);

  // for (int i = 0; i < 24; ++i) {
  // 	if (mychain->GetLeaf(Form("energy_layer%d", i))) {
  // 		setBranch(mychain, m_branches, Form("energy_layer%d", i), &m_total_layer_cell_energy[i]);
  // 	}
  // }
}

float TFCSAnalyzerBase::DeltaPhi( float phi1, float phi2 ) {
  double result = phi1 - phi2;

  while ( result > TMath::Pi() ) { result -= 2 * TMath::Pi(); }
  while ( result <= -TMath::Pi() ) { result += 2 * TMath::Pi(); }
  return result;
}

std::vector<float> TFCSAnalyzerBase::Getxbins( TH1F* histo, int nbins ) {
  // * calculate variable bin width in alpha and dr making sure each bin has almost equal amount of hits.

  bool isAlpha = true;

  std::vector<float> xbins;

  std::string title = histo->GetTitle();

  if ( title.compare( "h_dr" ) == 0 ) { isAlpha = false; }

  if ( m_debug ) { cout << "title = " << title.c_str() << " isAlpha = " << isAlpha << endl; }

  if ( isAlpha ) {
    xbins.push_back( TMath::Pi() / 8 );
  } else {
    xbins.push_back( 0 );
  }

  float AvgHitsPerBin = histo->Integral() / nbins;

  float hitCounts = 0;

  for ( int ibin = 1; ibin < histo->GetNbinsX() + 1; ibin++ ) {
    if ( hitCounts < AvgHitsPerBin ) {
      hitCounts = hitCounts + histo->GetBinContent( ibin );
    } else if ( hitCounts >= AvgHitsPerBin ) {
      xbins.push_back( histo->GetBinLowEdge( ibin ) + histo->GetBinWidth( ibin ) );
      hitCounts = 0;
    }
  }

  int   nRmax = histo->FindLastBinAbove( 0 );
  float Rmax  = histo->GetBinLowEdge( nRmax ) + histo->GetBinWidth( nRmax );

  if ( isAlpha ) {
    xbins.push_back( 2 * TMath::Pi() + TMath::Pi() / 8 );
  } else {
    xbins.push_back( Rmax );
  }

  return xbins;
}

double TFCSAnalyzerBase::GetBinUpEdge( TH1F* histo, float cutoff ) {

  int nbins = histo->GetNbinsX();

  double energy = 0.;
  if ( histo ) energy = histo->Integral();

  double threshold  = cutoff * energy;
  double tot_energy = 0.;
  double bin_edge   = 0.;

  if ( energy <= 0 ) {
    bin_edge = histo->GetXaxis()->GetBinUpEdge( 2 );
    return bin_edge;
  }

  for ( int i = 0; i < nbins; i++ ) {

    tot_energy += histo->GetBinContent( i );

    if ( tot_energy >= threshold ) {

      bin_edge = histo->GetXaxis()->GetBinUpEdge( i );
      break;
    }
  }

  return bin_edge;
}

bool TFCSAnalyzerBase::findWord( const std::string sentence, std::string search ) {

  bool has = false;
  // std::cout << "sentence : " << sentence << std::endl;

  size_t pos;
  pos = sentence.find( search );

  if ( pos != std::string::npos ) {
    has = true;
    // std::cout << "variable contains " << search << std::endl;
  } else {
    has = false;
  }

  return has;
}

std::string TFCSAnalyzerBase::replaceChar( std::string str, char find, char replace ) {
  for ( unsigned int i = 0; i < str.length(); ++i ) {
    if ( str[i] == find ) str[i] = replace;
  }

  return str;
}

void TFCSAnalyzerBase::MakeColorVector() {
  v_color.push_back( kBlue + 0 );
  v_color.push_back( kGreen + 2 );
  v_color.push_back( kViolet + 2 );
  v_color.push_back( kRed + 1 );
  v_color.push_back( kCyan + 3 );
  v_color.push_back( kOrange + 3 );
  v_color.push_back( kAzure + 6 );
  v_color.push_back( kYellow + 3 );
}

TString TFCSAnalyzerBase::GetLabel() {

  std::string particle = m_particle;

  if ( particle.compare( "electron" ) == 0 ) {
    particle = "e^ {#pm}";
  } else if ( particle.compare( "pionminus" ) == 0 ) {
    particle = "#pi^{-}";
  } else if ( particle.compare( "pionplus" ) == 0 ) {
    particle = "#pi^{+}";
  } else if ( particle.compare( "photon" ) == 0 ) {
    particle = "#gamma";
  } else if ( particle.compare( "pion" ) == 0 ) {
    particle = "#pi^{#pm}";
  }

  TString label = Form( " %i GeV, %s, %.2f < |#eta| < %.2f", (int)( m_energy / 1e3 ), particle.c_str(), m_etamin / 1e2,
                        m_etamax / 1e2 );

  return label;
}

TString TFCSAnalyzerBase::GetLayerName( int layerid ) {

  TString layer = "";
  if ( layerid == 0 )
    layer = "EM barrel presampler";
  else if ( 0 < layerid && layerid <= 3 )
    layer = Form( "EM barrel %i", layerid );
  else if ( layerid == 4 )
    layer = "EM endcap presampler";
  else if ( 4 < layerid && layerid <= 7 )
    layer = Form( "EM endcap %i ", layerid - 4 );
  else if ( 7 < layerid && layerid <= 11 )
    layer = Form( "Hadronic endcap %i ", layerid - 7 );
  else if ( 11 < layerid && layerid <= 14 )
    layer = Form( "Tile barrel %i ", layerid - 11 );
  else if ( 14 < layerid && layerid <= 17 )
    layer = Form( "Tile Gap %i ", layerid - 14 );
  else if ( 17 < layerid && layerid <= 20 )
    layer = Form( "Tile extended barrel %i ", layerid - 17 );
  else if ( 20 < layerid && layerid <= 23 )
    layer = Form( " Forwad EM endcap %i ", layerid - 20 );
  else
    layer = Form( "layer %i", layerid );

  return layer;
}

void TFCSAnalyzerBase::CreateHTML( std::string filename, std::vector<std::string> plotNames ) {

  std::string outDir = m_label;

  ofstream myhtml;
  myhtml.open( filename.c_str(), std::ios_base::trunc );
  TString label      = GetLabel();
  TString labeltitle = "Validation Plots: " + label;

  if ( myhtml.is_open() ) {
    myhtml << "<p align = \"center\" class=\"style3\"> Plots for : " << labeltitle << "</p>\n";
    myhtml << "<table align=\"center\" border=\"1\" cellspacing=\"0\" cellpadding=\"3\"><tr>\n";
    myhtml << "</tr><tr>\n";

    for ( unsigned iplot = 0; iplot < plotNames.size(); iplot++ ) {
      std::string plot = plotNames.at( iplot );

      myhtml << "<td align=\"center\" class=\"style2\" ><br> " << plot.c_str() << " </td>\n";
      myhtml << "<td align=\"center\" class=\"style1\" ><a  href=\"" << outDir.c_str() << "/"
             << ( plot + ".pdf" ).c_str() << "\"><img src=\"" << outDir.c_str() << "/" << ( plot + ".png" ).c_str()
             << "\" width=\"700\" height=\"600\"></a><br>"
             << ""
             << "</td>\n";

      myhtml << "</tr><tr>\n";
    }
    myhtml << "</body>\n";
    myhtml << "</html>\n";
    myhtml.close();

  } else {
    std::cout << "Error opening html file";
  }
}
