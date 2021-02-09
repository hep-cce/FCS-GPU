/*
  Copyright (C) 2002-2021 CERN for the benefit of the ATLAS collaboration
*/

#include "run_epara.cxx"
#include "runTFCS2DParametrizationHistogram.cxx"

void runTFCSEnergyShapeParametrizationChain( int pdgid = 22, int Emin = 64, int Emax = 4194304, double etamin = 0,
                                             double etamax = 5. ) {

  std::string sampleData = "../python/inputSampleList.txt";
  std::string topDir     = "/eos/atlas/user/a/ahasib/public/Simul-FastCalo/ParametrizationProductionVer04/";
  std::string topPlotDir = "/eos/user/a/ahasib/www/Simul-FastCalo/ParametrizationProductionVer04/";
  std::string version    = "ver04";

  int  npca1          = 5;
  int  npca2          = 1;
  bool run_validation = true;

  int   dsid_zv0         = -999;
  float energy_cutoff    = 0.9995;
  bool  do2DParam        = true;
  bool  isPhisymmetry    = true;
  bool  doMeanRz         = true;
  bool  useMeanRz        = false;
  bool  doZVertexStudies = false;

  int int_etamin = TMath::Nint( 100 * etamin );
  int int_etamax = TMath::Nint( 100 * etamax );

  std::vector<int> v_Energy;

  for ( int i = Emin; i <= Emax; i *= 2 ) { v_Energy.push_back( i ); }

  for ( int ieta = int_etamin; ieta < int_etamax; ieta += 5 ) {
    for ( int ienergy = 0; ienergy < v_Energy.size(); ienergy++ ) {
      int int_Emin = v_Energy.at( ienergy );

      std::string dsid =
          FCS_dsid::find_dsid( std::to_string( pdgid ), std::to_string( int_Emin ), std::to_string( ieta ), "0" );
      int int_dsid = std::stoi( dsid );

      run_epara( int_dsid, sampleData, topDir, npca1, npca2, run_validation, version, topPlotDir );

      runTFCS2DParametrizationHistogram( int_dsid, dsid_zv0, sampleData, topDir, version, energy_cutoff, topPlotDir,
                                         do2DParam, isPhisymmetry, doMeanRz, useMeanRz, doZVertexStudies );
    }
  }
}
