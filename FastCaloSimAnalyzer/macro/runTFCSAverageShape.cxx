/*
  Copyright (C) 2002-2019 CERN for the benefit of the ATLAS collaboration
*/

#include <docopt/docopt.h>

#include <TChain.h>
#include <TFile.h>
#include <TH2.h>
#include <TMath.h>
#include <TROOT.h>

#include "FastCaloSimAnalyzer/TFCSLateralShapeParametrizationHitChain.h"
#include <ISF_FastCaloSimEvent/TFCSParametrizationChain.h>
#include <ISF_FastCaloSimEvent/TFCSParametrizationEbinChain.h>

#include "FastCaloSimAnalyzer/TFCSAnalyzerHelpers.h"
#include "FastCaloSimAnalyzer/TFCSShapeValidation.h"
#include "FastCaloSimAnalyzer/TFCSValidationEnergy.h"
#include "FastCaloSimAnalyzer/TFCSValidationHitSpy.h"
#include "FastCaloSimAnalyzer/TFCSWriteCellsToTree.h"

#include "TFCSSampleDiscovery.h"

static const char* USAGE = R"(Compute average shape

Usage:
  runTFCSAverageShape [<pdgId>] [-s <seed> | --seed <seed>] [-e <int> | --eenergy <int>] [--etamin <float> --etamax <float>] [-p | --production]
  runTFCSAverageShape (-h | --help)

Options:
  -h --help                  Show help screen.
  -p --production            Run in production mode.
  -s <seed>, --seed <seed>   Random seed [default: 42].
  -e <int>, --energy <int>   Energy [default: 65536].
    --etamin <float>         Minimum eta [default: 0.2].
    --etamax <float>         Maximum eta [default: 0.25].
)";

int runTFCSAverageShape(int pdgid = 22, int int_E = 65536, double etamin = 0.2,
                        double etamax = 0.25, long seed = 42,
                        bool production = false) {
  FCS::LateralShapeParametrizationArray hit_to_cell_mapping = {
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  FCS::LateralShapeParametrizationArray numbers_of_hits = {
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

  FCS::init_hit_to_cell_mapping(hit_to_cell_mapping);
  FCS::init_numbers_of_hits(numbers_of_hits);

  // default_HistoShapeParametrization_number_of_hits = 500000;

  std::string particle = "";
  if (pdgid == 22) particle = "photon";
  if (pdgid == 211) particle = "pion";
  if (pdgid == 11) particle = "electron";
  std::string energy = Form("E%d", int_E);
  std::string eta = Form("eta%03d_%03d", TMath::Nint(etamin * 100),
                         TMath::Nint(etamax * 100));

  std::string prefix_E_eta =
      (particle + "_" + energy + "_" + eta + "_").c_str();
  std::string prefix_E_eta_title =
      particle + Form(", E=%d MeV, %4.2f<|#eta|<%4.2f", int_E, etamin, etamax);
  std::string prefixall = prefix_E_eta + "csall_allpca_";
  std::string prefixall_title = prefix_E_eta_title + ", all samples, all pca";

  auto sample = std::make_unique<TFCSSampleDiscovery>();
  int dsid = sample->findDSID(pdgid, int_E, etamin * 100, 0).dsid;
  FCS::SampleInfo sampleInfo = sample->findSample(dsid);

  std::string inputSample = sampleInfo.location;
  std::string shapefile = sample->getShapeName(dsid);
  std::string energyfile = sample->getSecondPCAName(dsid);
  std::string pcaSample = sample->getFirstPCAAppName(dsid);
  std::string avgSample = sample->getAvgSimShapeName(dsid);

#if defined(__linux__)
  std::cout << "* Running on linux system " << std::endl;
#endif

  TChain* inputChain = new TChain("FCS_ParametrizationInput");
  inputChain->Add(inputSample.c_str(), -1);

  int nentries = inputChain->GetEntries();

  std::cout << " * Prepare to run on: " << inputSample
            << " with entries = " << nentries << std::endl;

  TChain* pcaChain = new TChain("tree_1stPCA");
  pcaChain->Add(pcaSample.c_str());
  inputChain->AddFriend("tree_1stPCA");

  std::cout << " *   1stPCA: entries = " << pcaChain->GetEntries() << std::endl;

  auto fpca = std::unique_ptr<TFile>(TFile::Open(pcaSample.c_str()));
  if (!fpca) {
    std::cerr << "Error: Could not open file '" << pcaSample << "'"
              << std::endl;
    return 1;
  }

  std::vector<int> v_layer;

  TH2I* relevantLayers = (TH2I*)fpca->Get("h_layer");
  int npcabins = relevantLayers->GetNbinsX();
  for (int ibiny = 1; ibiny <= relevantLayers->GetNbinsY(); ibiny++) {
    if (relevantLayers->GetBinContent(1, ibiny) == 1)
      v_layer.push_back(ibiny - 1);
  }

  std::cout << " relevantLayers = ";
  for (auto i : v_layer) std::cout << i << " ";
  std::cout << " ; #pca bins = " << npcabins << std::endl;

  //////////////////////////////////////////////////////////
  ///// Creat validation steering
  //////////////////////////////////////////////////////////
  TFCSShapeValidation* analyze = new TFCSShapeValidation(inputChain, 0, seed);
  analyze->set_IsNewSample(true);
  analyze->set_Nentries(nentries);
  // analyze->set_Nentries(10000);
  analyze->set_Debug(0);

  // gROOT->ProcessLineSync(Form("TFCSShapeValidation*
  // analyze=(TFCSShapeValidation*)%p", analyze));

  std::cout << "=============================" << std::endl;
  //////////////////////////////////////////////////////////
  ///// Chain to read in the energies from the input file, then simulate the
  ///average shape from a histogram
  //////////////////////////////////////////////////////////
  TFCSParametrizationChain* runOriginalEnergyAvgShapeSim =
      new TFCSParametrizationChain(
          "original_Energy_sim_avgshape_histo",
          "original energy from input file, avg shape sim from histo");

  TFCSValidationEnergy* original_Energy = new TFCSValidationEnergy(
      "original_Energy", "original energy from input file", analyze);
  original_Energy->set_n_bins(npcabins);
  original_Energy->set_pdgid(pdgid);
  for (unsigned int i = 0; i < v_layer.size(); ++i)
    original_Energy->get_layers().push_back(v_layer[i]);
  runOriginalEnergyAvgShapeSim->push_back(original_Energy);

  TFCSParametrizationEbinChain* EbinChainAvgShape =
      FCS::NewShapeEbinCaloSampleChain(original_Energy, hit_to_cell_mapping,
                                       numbers_of_hits, shapefile, pdgid, int_E,
                                       etamin, etamax);
  runOriginalEnergyAvgShapeSim->push_back(EbinChainAvgShape);

  std::vector<TFCSValidationHitSpy*> hitspy_sim1(CaloCell_ID_FCS::MaxSample,
                                                 nullptr);
  std::vector<TFCSValidationHitSpy*> hitspy_sim2(CaloCell_ID_FCS::MaxSample,
                                                 nullptr);
  for (unsigned int i = 0; i < v_layer.size(); ++i) {
    int layer = v_layer[i];
    hitspy_sim1[layer] = new TFCSValidationHitSpy(
        Form("hitspy1_sample%d", layer),
        Form(
            "hitspy Nr.1 for sampling %d, original energy from input file, "
            "shape sim from histo",
            layer));
    hitspy_sim1[layer]->set_calosample(layer);
    hitspy_sim1[layer]->hist_hitgeo_dphi() = analyze->InitTH1(
        prefixall + Form("hist_hitspy1_sample%d_geodphi", layer), "1D", 256,
        -TMath::Pi() / 64, TMath::Pi() / 64, "dphi", "#hits");
    hitspy_sim1[layer]->hist_hitgeo_matchprevious_dphi() = analyze->InitTH1(
        prefixall + Form("hist_hitspy1_sample%d_geomatchprevious_dphi", layer),
        "1D", 256, -TMath::Pi() / 64, TMath::Pi() / 64, "dphi", "#hits");

    hitspy_sim2[layer] = new TFCSValidationHitSpy(
        Form("hitspy2_sample%d", layer),
        Form(
            "hitspy Nr.2 for sampling %d, original energy from input file, "
            "shape sim from histo",
            layer));
    hitspy_sim2[layer]->set_calosample(layer);
    hitspy_sim2[layer]->set_previous(hitspy_sim1[layer]);
    hitspy_sim2[layer]->hist_hitgeo_dphi() = analyze->InitTH1(
        prefixall + Form("hist_hitspy2_sample%d_geodphi", layer), "1D", 256,
        -TMath::Pi() / 64, TMath::Pi() / 64, "dphi", "#hits");
    hitspy_sim2[layer]->hist_hitgeo_matchprevious_dphi() = analyze->InitTH1(
        prefixall + Form("hist_hitspy2_sample%d_geomatchprevious_dphi", layer),
        "1D", 256, -TMath::Pi() / 64, TMath::Pi() / 64, "dphi", "#hits");
  }

  for (size_t i = 0; i < EbinChainAvgShape->size(); ++i) {
    if ((*EbinChainAvgShape)[i]
            ->InheritsFrom(TFCSLateralShapeParametrizationHitChain::Class())) {
      TFCSLateralShapeParametrizationHitChain* hitchain =
          (TFCSLateralShapeParametrizationHitChain*)(*EbinChainAvgShape)[i];
      int cs = hitchain->calosample();
      if (hitchain->size() > 0) {
        auto it = hitchain->chain().begin() + 1;
        hitchain->chain().insert(it, hitspy_sim1[cs]);
      }
      hitchain->push_back(hitspy_sim2[cs]);
    }
  }

  std::string filename_avgShape =
      production ? avgSample.c_str() : "AvgSimShape.root";
  auto file_avgshape = std::unique_ptr<TFile>(
      TFile::Open(filename_avgShape.c_str(), "RECREATE"));
  if (!file_avgshape) {
    std::cerr << "Error: Could not create file '" << filename_avgShape << "'"
              << std::endl;
    return 1;
  }

  TTree* tree_avgshape = nullptr;
  if (file_avgshape) {
    tree_avgshape = new TTree(Form("AvgShape"), Form("AvgShape"));
    TFCSWriteCellsToTree* tree_writer_AvgShapeSim = new TFCSWriteCellsToTree(
        "tree_writer_AvgShapeSim",
        "Tree writer for original energy from input file, shape sim from histo",
        tree_avgshape);
    runOriginalEnergyAvgShapeSim->push_back(tree_writer_AvgShapeSim);
    for (unsigned int i = 0; i < CaloCell_ID_FCS::MaxSample; ++i) {
      if (hitspy_sim1[i]) {
        file_avgshape->Add(hitspy_sim1[i]->hist_hitgeo_dphi());
        file_avgshape->Add(hitspy_sim1[i]->hist_hitgeo_matchprevious_dphi());
      }
      if (hitspy_sim2[i]) {
        file_avgshape->Add(hitspy_sim2[i]->hist_hitgeo_dphi());
        file_avgshape->Add(hitspy_sim2[i]->hist_hitgeo_matchprevious_dphi());
      }
    }
  }
  gROOT->cd();

#ifdef FCS_DEBUG
  runOriginalEnergyAvgShapeSim->Print();
  runOriginalEnergyAvgShapeSim->Print("short");
#endif

  analyze->add_validation("AvgShape", "Average shape sim",
                          runOriginalEnergyAvgShapeSim);
  std::cout << "=============================" << std::endl;

  //////////////////////////////////////////////////////////
  ///// Run over events
  //////////////////////////////////////////////////////////

  analyze->LoopEvents(-1);

  if (file_avgshape) {
    std::cout << "= Average Shape output tree =" << std::endl;
    file_avgshape->Write();
#ifdef FCS_DEBUG
    file_avgshape->ls();
// tree_avgshape->Print();
#endif
    std::cout << "=============================" << std::endl;

    file_avgshape->Close();
  }

  return 0;
}

int main(int argc, char** argv) {
  std::map<std::string, docopt::value> args =
      docopt::docopt(USAGE, {argv + 1, argv + argc}, true);

  int pdgId = args["<pdgId>"].isLong() ? args["<pdgId>"].asLong() : 22;
  bool production = args["--production"].asBool();
  long seed = args["--seed"].asLong();
  int E = args["--energy"].asLong();
  float etamin = std::stof(args["--etamin"].asString());
  float etamax = std::stof(args["--etamax"].asString());

  std::cout << "Running in production mode: " << production << std::endl;

  return runTFCSAverageShape(pdgId, E, etamin, etamax, seed, production);
}
