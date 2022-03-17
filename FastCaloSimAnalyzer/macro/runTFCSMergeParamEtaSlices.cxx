#include <docopt/docopt.h>

#include <TFile.h>
#include <TMath.h>
#include <TSystem.h>

#include "ISF_FastCaloSimEvent/TFCSInvisibleParametrization.h"
#include "ISF_FastCaloSimEvent/TFCSParametrizationAbsEtaSelectChain.h"
#include "ISF_FastCaloSimEvent/TFCSParametrizationPDGIDSelectChain.h"

static const char* USAGE = R"(Merge eta slices parametrizations into one file

Usage:
  runTFCSMergeParamEtaSlices [--emin <int>] [--emax <int>] [--etamin <float>] [--etamax <float>] [--bigParamFileName <string>] [-o <dir> | --output <dir>]
  runTFCSMergeParamEtaSlices (-h | --help)

Options:
  -h --help                    Show help screen.
  --emin <int>                 Minimum energy [default: 64].
  --emax <int>                 Maximum energy [default: 4194304].
  --etamin <float>             Minimum eta [default: 1.0].
  --etamax <float>             Maximum eta [default: 1.05].
  --bigParamFileName <string>  Name of the big param file [default: TFCSparam_v010.root]
  -o <dir>, --output <dir>     Output directory [default: TFCSParam].
)";

TFCSParametrizationBase* runTFCSMergeParamPDGIDEtaSlices(
    int pdgid, int int_Emin, int int_Emax, double etamin, double etamax,
    std::string outDir, TString bigParamFileName) {
  double Emin = int_Emin;
  double Emax = int_Emax;

  int int_etamin = TMath::Nint(100 * etamin);
  int int_etamax = TMath::Nint(100 * etamax);

  TString etapara_name = Form("SelEta_id%d_Mom%d_%d_eta_%d_%d", pdgid, int_Emin,
                              int_Emax, int_etamin, int_etamax);
  TString etapara_title =
      Form("Select eta for id=%d %3.1f<=Mom<%3.1f %4.2f<=|eta|<%4.2f", pdgid,
           Emin, Emax, etamin, etamax);
  TFCSParametrizationAbsEtaSelectChain* EtaSelectChain =
      new TFCSParametrizationAbsEtaSelectChain(etapara_name, etapara_title);
  EtaSelectChain->set_SplitChainObjects();

  for (; int_etamin < int_etamax; int_etamin += 5) {
    etamin = int_etamin / 100.0;
    etamax = (int_etamin + 5) / 100.0;

    int etamax_int = TMath::Nint(100 * etamax);

    TString Ekinpara_name = Form("SelEkin_id%d_Mom%d_%d_eta_%d_%d", pdgid,
                                 int_Emin, int_Emax, int_etamin, etamax_int);

    std::cout << " Ekinpara_name = " << Ekinpara_name.Data() << std::endl;

    TString fileName = Form("%s/%s", outDir.c_str(), Ekinpara_name.Data());

    auto file = std::unique_ptr<TFile>(TFile::Open(fileName + ".root"));
    if (!file) {
      std::cerr << "Error: Could not open file '" << fileName << "'"
                << std::endl;
      return nullptr;
    }
    file->ls();

    TFCSParametrizationBase* para =
        dynamic_cast<TFCSParametrizationBase*>(file->Get(Ekinpara_name));

    if (para) {
      EtaSelectChain->push_back_in_bin(para);
      file->Close();
      std::cout << "=========== EtaSelectChain ==========" << std::endl
                << "pdgid=" << pdgid << " " << int_Emin << "<=Mom<" << int_Emax
                << " " << etamin << "<=eta<" << etamax << std::endl;
      // para->Print("short");
      std::cout << "=====================================" << std::endl;
    } else {
      std::cout << "============= ERROR =================" << std::endl
                << "ERROR: pdgid=" << pdgid << " " << int_Emin << "<=Mom<"
                << int_Emax << " " << etamin << "<=eta<" << etamax << std::endl
                << "=====================================" << std::endl;
    }
  }

  return (TFCSParametrizationBase*)EtaSelectChain;
}

int runTFCSMergeParamEtaSlices(int int_Emin, int int_Emax, double etamin,
                               double etamax, std::string outDir,
                               std::string bigParamFileName) {
  TFCSParametrizationBase* para_photon = runTFCSMergeParamPDGIDEtaSlices(
      22, int_Emin, int_Emax, etamin, etamax, outDir, bigParamFileName);
  TFCSParametrizationBase* para_electron = runTFCSMergeParamPDGIDEtaSlices(
      11, int_Emin, int_Emax, etamin, etamax, outDir, bigParamFileName);
  TFCSParametrizationBase* para_pion = runTFCSMergeParamPDGIDEtaSlices(
      211, int_Emin, int_Emax, etamin, etamax, outDir, bigParamFileName);
  para_pion->set_match_all_pdgid();  /// match to all pdgid

  TFCSParametrizationPDGIDSelectChain* fullchain =
      new TFCSParametrizationPDGIDSelectChain("SelPDGID", "Select PDGID");
  fullchain->set_SimulateOnlyOnePDGID();

  TFCSInvisibleParametrization* invisible = new TFCSInvisibleParametrization(
      "RemoveInvisiblesMuons", "Do not simulate invisibles and muons");
  // Muons
  invisible->add_pdgid(13);
  invisible->add_pdgid(-13);
  // Neutrinos
  invisible->add_pdgid(12);
  invisible->add_pdgid(-12);
  invisible->add_pdgid(14);
  invisible->add_pdgid(-14);
  invisible->add_pdgid(16);
  invisible->add_pdgid(-16);
  fullchain->push_back(invisible);

  fullchain->push_back(para_photon);
  fullchain->push_back(para_electron);
  /// pions needs to be last since it will match invisibles, photons, electrons
  /// first and the rest is added to pions
  fullchain->push_back(para_pion);

  std::cout << "=============================================" << std::endl
            << "==== Finished adding param to full chain ====" << std::endl
            << "=============================================" << std::endl;

  const float toMB = 1.f / 1024.f;
  double tot_res_mem = 0.;
  double tot_virt_mem = 0.;

  double tot_res_mem_before = 0.;
  double tot_virt_mem_before = 0.;

  double tot_res_mem_after = 0.;
  double tot_virt_mem_after = 0.;

  auto fullchainfile =
      std::unique_ptr<TFile>(TFile::Open(bigParamFileName.c_str(), "recreate"));
  if (!fullchainfile) {
    std::cerr << "Error: Could not create file '" << bigParamFileName << "'"
              << std::endl;
    return 1;
  }

  fullchain->Write();
  fullchainfile->ls();

  static ProcInfo_t info;
  gSystem->GetProcInfo(&info);
  tot_res_mem_before += info.fMemResident * toMB;
  tot_virt_mem_before += info.fMemVirtual * toMB;

  std::cout << "======================================" << std::endl
            << "==== Now re-reading the full file ====" << std::endl
            << "======================================" << std::endl;
  TFCSParametrizationBase* testread =
      (TFCSParametrizationBase*)fullchainfile->Get(fullchain->GetName());
  gSystem->GetProcInfo(&info);
  tot_res_mem_after += info.fMemResident * toMB;
  tot_virt_mem_after += info.fMemVirtual * toMB;
  tot_res_mem = tot_res_mem_after - tot_res_mem_before;
  tot_virt_mem = tot_virt_mem_after - tot_virt_mem_before;

  testread->Print("short");

  fullchainfile->Close();

  std::cout << " -----------------------------------------" << std::endl;
  printf(" res  memory  = %g Mbytes\n", tot_res_mem);
  printf(" vir  memory = %g Mbytes\n", tot_virt_mem);
  std::cout << " -----------------------------------------" << std::endl;

  return 0;
}

int main(int argc, char** argv) {
  std::map<std::string, docopt::value> args =
      docopt::docopt(USAGE, {argv + 1, argv + argc}, true);

  int Emin = args["--emin"].asLong();
  int Emax = args["--emax"].asLong();
  float etamin = std::stof(args["--etamin"].asString());
  float etamax = std::stof(args["--etamax"].asString());
  std::string output = args["--output"].asString();
  std::string bigParamFileName = args["--bigParamFileName"].asString();

  return runTFCSMergeParamEtaSlices(Emin, Emax, etamin, etamax, output,
                                    bigParamFileName);
}
