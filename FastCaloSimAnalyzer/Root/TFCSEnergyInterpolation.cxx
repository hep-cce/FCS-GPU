/*
  Copyright (C) 2002-2018 CERN for the benefit of the ATLAS collaboration
*/

#include "FastCaloSimAnalyzer/TFCSEnergyInterpolation.h"
#include "ISF_FastCaloSimEvent/TFCSExtrapolationState.h"
#include "ISF_FastCaloSimEvent/TFCSSimulationState.h"
#include "ISF_FastCaloSimEvent/TFCSTruthState.h"
#include "TAxis.h"
#include "TCanvas.h"
#include "TGraph.h"
#include <iostream>

//=============================================
//======= TFCSEnergyInterpolation =========
//=============================================

TFCSEnergyInterpolation::TFCSEnergyInterpolation( const char* name, const char* title )
    : TFCSParametrization( name, title ), m_slope( 1 ), m_offset( 0 ) {}

FCSReturnCode TFCSEnergyInterpolation::simulate( TFCSSimulationState& simulstate, const TFCSTruthState* truth,
                                                 const TFCSExtrapolationState* ) {
  float Emean = m_slope * truth->Ekin() + m_offset;

  ATH_MSG_DEBUG( "set E=" << Emean << " for true Ekin=" << truth->Ekin() );
  simulstate.set_E( Emean );

  return FCSSuccess;
}

void TFCSEnergyInterpolation::Print( Option_t* option ) const {
  TString opt( option );
  bool    shortprint = opt.Index( "short" ) >= 0;
  bool    longprint  = msgLvl( MSG::DEBUG ) || ( msgLvl( MSG::INFO ) && !shortprint );
  TString optprint   = opt;
  optprint.ReplaceAll( "short", "" );
  TFCSParametrization::Print( option );

  if ( longprint ) ATH_MSG_INFO( optprint << "  Emean=" << m_slope << "*Ekin(true) + " << m_offset );
}

void TFCSEnergyInterpolation::unit_test( TFCSSimulationState* simulstate, TFCSTruthState* truth,
                                         const TFCSExtrapolationState* extrapol ) {
  if ( !simulstate ) simulstate = new TFCSSimulationState();
  if ( !truth ) truth = new TFCSTruthState();
  if ( !extrapol ) extrapol = new TFCSExtrapolationState();

  TFCSEnergyInterpolation test( "testTFCSEnergyInterpolation", "test TFCSEnergyInterpolation" );
  test.set_pdgid( 22 );
  test.set_Ekin_nominal( 1000 );
  test.set_Ekin_min( 1000 );
  test.set_Ekin_max( 100000 );
  test.set_eta_nominal( 0.225 );
  test.set_eta_min( 0.2 );
  test.set_eta_max( 0.25 );
  test.set_slope( 0.95 );
  test.set_offset( -50 );
  test.Print();

  truth->set_pdgid( 22 );

  TGraph* gr = new TGraph();
  gr->SetNameTitle( "testTFCSEnergyInterpolation", "test TFCSEnergyInterpolation" );
  gr->GetXaxis()->SetTitle( "Ekin [MeV]" );
  gr->GetYaxis()->SetTitle( "<E(reco)>/Ekin(true)" );

  int ip = 0;
  for ( float Ekin = 1000; Ekin <= 100000; Ekin *= 2 ) {
    // Init LorentzVector for truth. For photon Ekin=E
    truth->SetPxPyPzE( Ekin, 0, 0, Ekin );
    test.simulate( *simulstate, truth, extrapol );
    gr->SetPoint( ip, Ekin, simulstate->E() / Ekin );
    ++ip;
  }
  TCanvas* c = new TCanvas( "testTFCSEnergyInterpolation", "test TFCSEnergyInterpolation" );
  gr->Draw( "APL" );
  c->SetLogx();
}
