/*
  Copyright (C) 2002-2021 CERN for the benefit of the ATLAS collaboration
*/

#include <docopt/docopt.h>

#include <TFile.h>
#include <TMath.h>

#include "CLHEP/Random/TRandomEngine.h"

#include "ISF_FastCaloSimEvent/TFCSParametrizationBase.h"

#include "FastCaloSimAnalyzer/TFCSAnalyzerHelpers.h"

static const char* USAGE =
    R"(Create parametrization eta slice

Usage:
  runTFCSCreateParamEtaSlice [--pdgId <pdgId>] [-s <seed> | --seed <seed>] [--emin <int> --emax <int> --etamin <float>] [-o <dir> | --output <dir>]
  runTFCSCreateParamEtaSlice (-h | --help)

Options:
  -h --help                  Show help screen.
  --pdgId <pdgId>            Particle ID [default: 11].
  -s <seed>, --seed <seed>   Random seed [default: 42].
    --emin <int>             Minimum energy [default: 64].
    --emax <int>             Maximum energy [default: 4194304].
    --etamin <float>         Minimum eta [default: 3.0].
  -o <dir>, --output <dir>   Output directory [default: ./].
)";


int runTFCSCreateParamEtaSlice(int pdgid, int int_Mom_min, int int_Mom_max, double etamin, std::string outDir, long seed)
{

  system( ( "mkdir -p " + outDir ).c_str() );

  FCS::LateralShapeParametrizationArray hit_to_cell_mapping = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  FCS::LateralShapeParametrizationArray numbers_of_hits = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

  FCS::init_hit_to_cell_mapping( hit_to_cell_mapping, true );
  FCS::init_numbers_of_hits( hit_to_cell_mapping, 1 ); // 1 is nominal stochastic term for EM

  // double Emin = int_Emin;
  // double Emax = int_Emax;
  int int_etamin = TMath::Nint( 100 * etamin );
  // int int_etamax = TMath::Nint(100 * etamin) + 5;
  double etamax = ( int_etamin + 5 ) / 100.0;

  CLHEP::TRandomEngine* randEngine = new CLHEP::TRandomEngine();
  randEngine->setSeed( seed );


  TFCSParametrizationBase* para = FCS::NewEnergyChain(randEngine, hit_to_cell_mapping, numbers_of_hits, pdgid, int_Mom_min, int_Mom_max, etamin, etamax);
  if ( para ) {
    TString filename = Form( "%s/%s", outDir.c_str(), para->GetName() );
    auto    file     = std::unique_ptr<TFile>( TFile::Open( filename + ".root", "recreate" ) );
    if ( !file ) {
      std::cerr << "Error: Could not create file '" << filename + ".root" << "'" << std::endl;
      return 1;
    }
    para->Write();
    file->ls();
    file->Close();
  }

  if ( para ) delete para;

  return 0;
}

int main(int argc, char **argv)
{
  std::map<std::string, docopt::value> args
    = docopt::docopt(USAGE, {argv + 1, argv + argc}, true);

  int         pdgId  = args["--pdgId"].asLong();
  long        seed   = args["--seed"].asLong();
  int         Emin   = args["--emin"].asLong();
  int         Emax   = args["--emax"].asLong();
  float       etamin = std::stof( args["--etamin"].asString() );
  std::string output = args["--output"].asString();

  return runTFCSCreateParamEtaSlice( pdgId, Emin, Emax, etamin, output, seed );
}
