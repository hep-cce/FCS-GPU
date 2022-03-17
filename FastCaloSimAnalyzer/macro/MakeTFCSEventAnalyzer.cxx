/*
  Copyright (C) 2002-2017 CERN for the benefit of the ATLAS collaboration
*/

#include "TFCSfirstPCA.h"
#include "TFCSFlatNtupleMaker.h"
#include "TFCSInputValidationPlots.h"
#include "TFCS2DParametrization.h"

#include "TFile.h"
#include "TChain.h"
#include "TTree.h"
#include "TString.h"
#include "TH2.h"

#include <iostream>
#include <algorithm>
#include <string>
#include <stdlib.h>
#include <tuple>

void MakeTFCSEventAnalyzer();

// int main(int argc, char const *argv[])
void MakeTFCSEventAnalyzer() {

  std::string particle = "photon";
  // particle = "pionplus";
  // particle = "electron";
  std::string energy = "E65536";
  // energy = "E50000";
  std::string eta = "eta020_025";
  // eta = "eta205_205"
  std::string production = "InputSamplesLocalProd2017";
  std::string release = "rel_21_0_47";
  std::string campaign = "mc16_13TeV";
  std::string merge = "merged_1mm";
  // merge = "merged_2mm_z0timeshift";

  std::string topDir = "";
  std::string plotDir = "";

#if defined(__linux__)
  std::cout << "* Running on linux system " << std::endl;
  topDir = "/eos/atlas/atlascerngroupdisk/proj-simul/";
  plotDir = "/eos/project/a/atlas-fastcalosim/www/";
#endif

#if defined(__APPLE__)
  std::cout << "* Running on mac os system " << std::endl;
  topDir = "/Users/ahasib/Documents/Analysis/Data/FCS/";
  plotDir = "/Users/ahasib/Documents/Analysis/Data/FCS/";
#endif

  std::string label = particle + "." + energy + "." + eta;
  TString Sample = topDir + production + "/" + release + "/%s/" + campaign +
                   "." + label + "." + merge + ".%s";

  TString inputSample(Form(Sample, "Samples", "calohit.root"));
  TString pcaSample(Form(Sample, "PCAs", "firstPCA.root"));
  TString flatSample(Form(Sample, "flatSamples", "flatcalohit.root"));
  TString shapeParaFile(Form(Sample, "shapePara", "shapepara.root"));

  string pcaDir = topDir + production + "/" + release + "/PCAs/";
  string flatSampleDir = topDir + production + "/" + release + "/flatSamples/";
  string InputPlotsDir = plotDir + production + "/" + release +
                         "/InputValidationPlots/" + label + "/";
  string shapeParaDir = topDir + production + "/" + release + "/shapePara/";

  system(("mkdir -p " + pcaDir).c_str());
  system(("mkdir -p " + flatSampleDir).c_str());
  system(("mkdir -p " + InputPlotsDir).c_str());
  system(("mkdir -p " + shapeParaDir).c_str());

  std::string energy_label = energy.erase(0, 1);
  int part_energy = stoi(energy_label);
  // std::cout << " energy = " << part_energy << std::endl;

  std::string eta_label = eta.erase(0, 3);
  // std::cout << " eta_label = " << eta_label << std::endl;

  std::string etamin_label = eta_label.substr(0, eta_label.find("_"));
  std::string etamax_label = eta_label.substr(4, eta_label.find("_"));

  float etamin = atof(etamin_label.c_str());
  float etamax = atof(etamax_label.c_str());

  // std::cout << "eta min, max = " << etamin << ", " << etamax << std::endl ;

  TChain* inputChain = new TChain("FCS_ParametrizationInput");
  inputChain->Add(inputSample);

  int nentries = inputChain->GetEntries();
  // nentries = 1000;

  std::cout << " * Prepare to run on: " << inputSample
            << " with entries = " << nentries << std::endl;

  TFCSfirstPCA* myfirstPCA = new TFCSfirstPCA(inputChain, pcaSample);
  myfirstPCA->set_cumulativehistobins(5000);
  myfirstPCA->set_edepositcut(0.001);
  myfirstPCA->apply_etacut(
      0);  // this flag is for the old files, which are already sliced in eta
  myfirstPCA->set_etacut(0.2, 0.25);
  myfirstPCA->set_pcabinning(5, 1);
  // myfirstPCA->run();

  std::cout << " Finished generating PCA file " << std::endl;

  TChain* pcaChain = new TChain("tree_1stPCA");
  pcaChain->Add(pcaSample);
  inputChain->AddFriend("tree_1stPCA");

  TFile* fpca = TFile::Open(pcaSample);
  std::vector<int> v_layer;

  TH2I* relevantLayers = (TH2I*)fpca->Get("h_layer");
  for (int ibiny = 1; ibiny <= relevantLayers->GetNbinsY(); ibiny++) {
    if (relevantLayers->GetBinContent(1, ibiny) == 1)
      v_layer.push_back(ibiny - 1);
  }

  std::cout << " relevantLayers = ";
  for (auto i : v_layer) std::cout << i << " ";
  std::cout << "\n";

  TFCSFlatNtupleMaker* analyze =
      new TFCSFlatNtupleMaker(inputChain, flatSample, v_layer);
  analyze->set_IsNewSample(true);
  analyze->set_Nentries(nentries);
  analyze->set_label(label);
  analyze->set_merge(merge);
  analyze->set_particle(particle);
  analyze->set_energy(part_energy);
  analyze->set_eta(etamin, etamax);
  analyze->LoopEvents();
  // analyze->StudyHitMerging();

  TFile* fmini = TFile::Open(flatSample);
  TTree* inputTree = (TTree*)fmini->Get("FCS_flatNtuple");
  std::cout << " * Running on " << flatSample
            << " with entries =" << inputTree->GetEntries() << std::endl;
  std::string html = InputPlotsDir + "../" + label + ".html";

  std::vector<std::tuple<std::string, std::string, std::string>> histos = {
      {"d_phi_mm:d_eta_mm", "r*#delta#eta [mm]", "r*#delta#phi [mm]"},
      {"d_eta_mm", "r*#delta#eta [mm]", ""},
      {"d_phi_mm", "r*#delta#phi [mm]", ""}, {"radius_mm", "r [mm]", ""},
      {"alpha_mm", "#alpha [mm]", ""}
      // {"d_phi:d_eta", "#delta#eta", "#delta#phi"},
      // {"d_eta", "#delta#eta", ""},
      // {"d_phi", "#delta#phi", ""},
      // {"radius", "r", ""},
      // {"alpha", "#alpha", ""}
  };

  // TFCSInputValidationPlots* validate = new
  // TFCSInputValidationPlots(inputTree, InputPlotsDir, v_layer);

  // validate->set_label(label);
  // validate->set_particle(particle);
  // validate->set_energy(part_energy);
  // validate->set_eta(etamin, etamax);
  // validate->CreateBinning(0.9995); // fraction of the total energy

  // std::vector<std::string> v_hist;
  // for (unsigned int ihist = 0; ihist < histos.size(); ihist++)
  // {
  //     std::string hist = std::get<0>(histos[ihist]);
  //     v_hist.push_back(hist);
  //     std::string xlabel = std::get<1>(histos[ihist]);
  //     std::string ylabel = std::get<2>(histos[ihist]);

  //     if (validate->findWord(hist, ":"))
  //     {
  //         validate->PlotTH2(hist, xlabel, ylabel);
  //     } else
  //     {
  //         validate->PlotTH1(hist, xlabel);
  //     }
  // }

  // validate->CreateInputValidationHTML(html, v_hist);

  // TFCS2DParametrization* shapepara = new TFCS2DParametrization(inputTree,
  // shapeParaFile.Data(), v_layer);
  // shapepara->CreateShapeHistograms(0.9995, "default");

  // return 1;
}
