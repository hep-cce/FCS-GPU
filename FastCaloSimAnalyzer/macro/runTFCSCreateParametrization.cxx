/*
  Copyright (C) 2002-2023 CERN for the benefit of the ATLAS collaboration
*/

#include <docopt/docopt.h>

#include <TH2.h>
#include <TMath.h>
#include <TSystem.h>

#include "CLHEP/Random/TRandomEngine.h"

#include "ISF_FastCaloSimEvent/TFCSInvisibleParametrization.h"
#include "FastCaloSimAnalyzer/TFCSLateralShapeParametrizationHitChain.h"
#include "ISF_FastCaloSimEvent/TFCSParametrizationAbsEtaSelectChain.h"
#include "ISF_FastCaloSimEvent/TFCSParametrizationEbinChain.h"
#include "ISF_FastCaloSimEvent/TFCSParametrizationPDGIDSelectChain.h"
#include "ISF_FastCaloSimEvent/TFCSPCAEnergyParametrization.h"
#include "ISF_FastCaloSimEvent/TFCSSimulationState.h"

#include "FastCaloSimAnalyzer/TFCSAnalyzerBase.h"
#include "FastCaloSimAnalyzer/TFCSAnalyzerHelpers.h"

#include "TFCSSampleDiscovery.h"

using namespace std;


static const char * USAGE =
    R"(Create one big parametrization file from individual energy and shape parametrizations

Usage:
  runTFCSCreateParametrization [<pdgId>] [-s <seed> | --seed <seed>] [--emin <int> --emax <int> --etamin <float> --etamax <float>] [-o <dir> | --output <dir>]
  runTFCSCreateParametrization (-h | --help)

Options:
  -h --help                  Show help screen.
  -s <seed>, --seed <seed>   Random seed [default: 42].
    --emin <int>             Minimum energy [default: 64].
    --emax <int>             Maximum energy [default: 4194304].
    --etamin <float>         Minimum eta [default: 1.0].
    --etamax <float>         Maximum eta [default: 1.05].
  -o <dir>, --output <dir>   Output directory [default: TFCSParam].
)";


// TFCSEnergyParametrization* NewPCAEnergyParametrization(CLHEP::HepRandomEngine *randEngine, std::string filename, int pdgid, int E, double etamin, double etamax)
// {
//   double Emin = (E / 2) * TMath::Sqrt(2);
//   double Emax = E * TMath::Sqrt(2);
//   return NewPCAEnergyParametrization(randEngine, filename, pdgid, E, Emin, Emax, etamin, etamax);
// }

TFCSParametrizationBase* NewEtaChain(CLHEP::HepRandomEngine *randEngine, const FCS::LateralShapeParametrizationArray &mapping, const FCS::LateralShapeParametrizationArray &numbersOfHits, int pdgid, int int_Mom_min, int int_Mom_max, double etamin, double etamax)
{
 
  int int_etamin = TMath::Nint(100 * etamin);
  int int_etamax = TMath::Nint(100 * etamax);

  TString etapara_name = Form("SelEta_id%d_Mom%d_%d_eta_%d_%d", pdgid, int_Mom_min, int_Mom_max, int_etamin, int_etamax);
  TString etapara_title = Form("Select eta for id=%d %d<=Mom<%d %4.2f<=|eta|<%4.2f", pdgid, int_Mom_min, int_Mom_max, etamin, etamax);
  TFCSParametrizationAbsEtaSelectChain* EtaSelectChain = new TFCSParametrizationAbsEtaSelectChain(etapara_name, etapara_title);

  for (; int_etamin < int_etamax; int_etamin += 5) {
    etamin = int_etamin / 100.0;
    etamax = (int_etamin + 5) / 100.0;
    TFCSParametrizationBase* para = FCS::NewEnergyChain(randEngine, mapping, numbersOfHits, pdgid, int_Mom_min, int_Mom_max, etamin, etamax);
    if (para) {
      EtaSelectChain->push_back_in_bin(para);
      cout << "=========== EtaSelectChain ==========" << endl;
      cout << "pdgid=" << pdgid << " " << int_Mom_min << "<=Mom<" << int_Mom_max << " " << etamin << "<=eta<" << etamax << endl;
      //para->Print("short");
      cout << "=====================================" << endl;
    } else {
      cout << "============= ERROR =================" << endl;
      cout << "ERROR: pdgid=" << pdgid << " " << int_Mom_min << "<=Mom<" << int_Mom_max << " " << etamin << "<=eta<" << etamax << endl;
      cout << "=====================================" << endl;
    }
  }

  return (TFCSParametrizationBase*)EtaSelectChain;
}

int runTFCSCreateParametrization(int pid = 22, int Emin = 64, int Emax = 4194304, float etamin = 1.0, float etamax = 1.05, std::string topDir = "TFCSParam", long seed = 42)
{
  FCS::LateralShapeParametrizationArray hit_to_cell_mapping = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  FCS::LateralShapeParametrizationArray numbers_of_hits = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

  FCS::init_hit_to_cell_mapping(hit_to_cell_mapping);
  FCS::init_numbers_of_hits(numbers_of_hits, 1); // 1 is nominal stochastic term for EM


  system(("mkdir -p " + topDir).c_str());


  CLHEP::TRandomEngine *randEngine = new CLHEP::TRandomEngine();
  randEngine->setSeed(seed);


  // TFCSParametrizationBase* para_photon = NewParametrization(randEngine, "/eos/atlas/user/a/ahasib/public/Simul-FastCalo/ParametrizationPhoton/ver02/mc16_13TeV.431004.ParticleGun_pid22_E65536_disj_eta_m25_m20_20_25_zv_0.secondPCA.ver02.root", "/eos/atlas/user/a/ahasib/public/Simul-FastCalo/ParametrizationPhoton/ver02/mc16_13TeV.431004.ParticleGun_pid22_E65536_disj_eta_m25_m20_20_25_zv_0.shapepara.ver02.root", 22, 65536, 0.2, 0.25);



  TFCSParametrizationBase* param = NewEtaChain(randEngine, hit_to_cell_mapping, numbers_of_hits, pid, Emin, Emax, etamin, etamax);

  TFCSParametrizationPDGIDSelectChain* fullchain = new TFCSParametrizationPDGIDSelectChain("SelPDGID", "Select PDGID");
  fullchain->set_SimulateOnlyOnePDGID();
  TFCSInvisibleParametrization* invisible = new TFCSInvisibleParametrization("RemoveInvisiblesMuons", "Do not simulate invisibles and muons");
  //Muons
  invisible->add_pdgid(13);
  invisible->add_pdgid(-13);
  //Neutrinos
  invisible->add_pdgid(12);
  invisible->add_pdgid(-12);
  invisible->add_pdgid(14);
  invisible->add_pdgid(-14);
  invisible->add_pdgid(16);
  invisible->add_pdgid(-16);
  fullchain->push_back(invisible);

  fullchain->push_back(param);

  fullchain->Print();
  cout << "==================================" << endl;
  fullchain->Print("short");

  TString paramFileName = Form("%s/%s", topDir.c_str(), param->GetName());

  auto fullchainfile = std::unique_ptr<TFile>(TFile::Open(paramFileName, "recreate"));
  if (!fullchainfile) {
    std::cerr << "Error: Could not create file '" << paramFileName << "'" << std::endl;
    return 1;
  }

  fullchain->Write();
  //para_photon_simple->Write();
  //para_pion_simple->Write();
  fullchainfile->ls();
  fullchainfile->Close();


  static ProcInfo_t info;
  const float toMB = 1.f / 1024.f;

  fullchainfile = std::unique_ptr<TFile>(TFile::Open(paramFileName));
  if (!fullchainfile) {
    std::cerr << "Error: Could not open file '" << paramFileName << "'" << std::endl;
    return 1;
  }

  gSystem->GetProcInfo(&info);
  printf(" res  memory = %g Mbytes\n", info.fMemResident * toMB);
  printf(" vir  memory = %g Mbytes\n", info.fMemVirtual * toMB);
  printf(" Now load photon param:\n");
  TFCSParametrizationBase* fullpara = (TFCSParametrizationBase*)fullchainfile->Get("SelPDGID");
  gSystem->GetProcInfo(&info);
  printf(" res  memory = %g Mbytes\n", info.fMemResident * toMB);
  printf(" vir  memory = %g Mbytes\n", info.fMemVirtual * toMB);
  fullchainfile->Close();

  if (param) delete param;
  if (fullchain) delete fullchain;
  if (invisible) delete invisible;

  return 0;
}

int main(int argc, char **argv)
{
  std::map<std::string, docopt::value> args
    = docopt::docopt(USAGE, {argv + 1, argv + argc}, true);

  int pdgId = args["<pdgId>"].isLong() ? args["<pdgId>"].asLong() : 22;
  long seed = args["--seed"].asLong();
  int Emin = args["--emin"].asLong();
  int Emax = args["--emax"].asLong();
  float etamin = std::stof(args["--etamin"].asString());
  float etamax = std::stof(args["--etamax"].asString());
  std::string output = args["--output"].asString();

  return runTFCSCreateParametrization(pdgId, Emin, Emax, etamin, etamax, output, seed);
}
