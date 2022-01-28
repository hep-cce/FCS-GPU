/*
  Copyright (C) 2002-2021 CERN for the benefit of the ATLAS collaboration
*/

#include "TH1D.h"
#include "TFile.h"
#include "TROOT.h"
#include "TSystem.h"
#include "TApplication.h"
#include "TTree.h"
#include "TSystem.h"
#include "TMath.h"
#include "EnergyParametrizationValidation.h"

#include <iostream>

using namespace std;

void EnergyParametrizationValidation::autozoom( TH1D* h1, double& min, double& max, double& rmin, double& rmax ) {

  double min1, max1;
  min1 = h1->GetXaxis()->GetXmin();
  max1 = h1->GetXaxis()->GetXmax();

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

TH1D* EnergyParametrizationValidation::refill( TH1D* h_in, double min, double max, double rmin, double rmax ) {

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
        delete h_clone;
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

  TH1D* h_out = new TH1D( "h_out", "h_out", Nbins, rmin, rmax );
  for ( int b = 1; b <= h_out->GetNbinsX(); b++ ) {
    h_out->SetBinContent( b, h_in->GetBinContent( start + b ) );
    h_out->SetBinError( b, h_in->GetBinError( start + b ) );
  }

  // if(debug) cout<<"AFTER refill ->underflow "<<h_out->GetBinContent(0)<<" startbin "<<start<<" minimum "<<min<<endl;

  return h_out;
}
