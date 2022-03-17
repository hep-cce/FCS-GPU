/*
  Copyright (C) 2002-2017 CERN for the benefit of the ATLAS collaboration
*/

#include "TFCSFlatNtupleMaker.h"
#include "TFCSfirstPCA.h"

#include "TFile.h"
#include "TChain.h"
#include "TTree.h"
#include "TString.h"
#include "TH2.h"

#include <iostream>
#include <algorithm>
#include <string>
#include <stdlib.h>

void MakeTFCSNtupleMaker();

// int main(int argc, char const *argv[])
void MakeTFCSNtupleMaker() {

  std::string particle = "photon";
  std::string energy = "E50000";
  std::string eta = "eta020_025";

  std::string label = particle + "." + energy + "." + eta;

  std::string topDir = "";

#if defined(__linux__)
  std::cout << "* Running on linux system " << std::endl;
  topDir =
      "/eos/atlas/atlascerngroupdisk/proj-simul/InputSamplesLocalProd2017/"
      "rel_21_0_42/Samples/";
#endif

#if defined(__APPLE__)
  std::cout << "* Running on mac os system " << std::endl;
  topDir =
      "/Users/ahasib/Documents/Analysis/Data/FCS/InputSamplesLocalProd2017/"
      "rel_21_0_40/SpecialSamples/";
#endif

  TString inputSample =
      topDir + "photon_E50000_eta_20_25.calohit_all_1mm_noParamAlg.root";

  TString pcaSample =
      topDir + "photon_E50000_eta_20_25.pca_all_1mm_noParamAlg.root";

  TString flatSample =
      topDir + "photon_E50000_eta_20_25.flatcalohit_all_1mm_noParamAlg.root";

  TChain *inputChain = new TChain("FCS_ParametrizationInput");
  inputChain->Add(inputSample);

  int nentries = inputChain->GetEntries();

  std::cout << " * Prepare to run on: " << inputSample
            << " with entries = " << nentries << std::endl;

  // TFCSfirstPCA *myfirstPCA = new TFCSfirstPCA(inputChain, pcaSample);
  // myfirstPCA->set_cumulativehistobins(5000);
  // myfirstPCA->set_edepositcut(0.001);
  // myfirstPCA->apply_etacut(0); //this flag is for the old files, which are
  // already sliced in eta
  // myfirstPCA->set_etacut(0.2, 0.25);
  // myfirstPCA->set_pcabinning(5, 1);
  // myfirstPCA->run();

  // std::cout << " Finished generating PCA file " << std::endl;

  TChain *pcaChain = new TChain("tree_1stPCA");
  pcaChain->Add(pcaSample);
  inputChain->AddFriend("tree_1stPCA");

  TFile *fpca = TFile::Open(pcaSample);
  std::vector<int> v_layer;

  // v_layer.push_back(2);

  TH2I *relevantLayers = (TH2I *)fpca->Get("h_layer");
  for (int ibiny = 1; ibiny <= relevantLayers->GetNbinsY(); ibiny++) {
    if (relevantLayers->GetBinContent(1, ibiny) == 1)
      v_layer.push_back(ibiny - 1);
  }

  std::cout << " relevantLayers = ";
  for (auto i : v_layer) std::cout << i << " ";
  std::cout << "\n";

  TFCSFlatNtupleMaker *analyze =
      new TFCSFlatNtupleMaker(inputChain, flatSample, v_layer);
  analyze->set_IsNewSample(true);
  analyze->set_Nentries(nentries);
  analyze->LoopEvents();

  // return 1;
}
