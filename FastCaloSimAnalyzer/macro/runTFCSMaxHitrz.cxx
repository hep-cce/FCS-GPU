/*
  Copyright (C) 2002-2018 CERN for the benefit of the ATLAS collaboration
*/

// #include "TFCSfirstPCA.h"
#include "TFCSAnalyzerBase.h"
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

string prefixlayer;
string prefixEbin;
string prefixall;

TH2* Create2DHistogram(TH2* h, float energy_cutoff) {

  TH1F* h1 = (TH1F*)h->ProjectionY();

  float ymin = h->GetYaxis()->GetBinLowEdge(1);
  float ymax = TFCSAnalyzerBase::GetBinUpEdge(h1, energy_cutoff);
  float binwidthx = h->GetYaxis()->GetBinWidth(1);
  int nbinsy = (int)((ymax - ymin) / binwidthx);

  float xmin = h->GetXaxis()->GetXmin();
  float xmax = h->GetXaxis()->GetXmax();
  int nbinsx = h->GetXaxis()->GetNbins();

  std::string title = h->GetTitle();

  TH2* h2 = new TH2F(title.c_str(), title.c_str(), nbinsx, xmin, xmax, nbinsy,
                     ymin, ymax);

  for (auto j = 0; j <= h2->GetNbinsY(); ++j) {
    for (auto i = 0; i <= h2->GetNbinsX(); ++i) {
      h2->SetBinContent(i, j, h->GetBinContent(i, j));
    }
  }

  float avg = h2->Integral(1, nbinsx, 1, 1) / nbinsx;

  for (int i = 1; i < nbinsx + 1; i++) h2->SetBinContent(i, 1, avg);

  h2->Scale(1 / h2->Integral());

  return h2;
}

void CreatePolarPlot(TH2F* h, std::string outDir) {

  gROOT->SetBatch(1);

  system(("mkdir -p " + outDir).c_str());

  std::string title = h->GetTitle();
  std::string xlabel = "x [mm]";
  std::string ylabel = "y [mm]";
  std::string zlabel = "Energy normalized to unity";

  TCanvas* c =
      TFCSAnalyzerBase::PlotPolar(h, title.c_str(), xlabel, ylabel, zlabel, 4);

  std::string outfile = outDir + title;

  c->SaveAs((outfile + ".png").c_str());

  delete c;
}

void runTFCSMaxHitrz(int dsid = 431004,
                     std::string sampleData = "../python/inputSampleList.txt",
                     std::string topDir = "output/",
                     std::string version = "ver01",
                     float energy_cutoff = 0.9995, bool isPhisymmetry = true,
                     std::string topPlotDir = "output_plot/") {

  system(("mkdir -p " + topDir).c_str());

  /////////////////////////////
  // read smaple information
  // based on DSID
  //////////////////////////

  TFCSAnalyzerBase::SampleInfo sample;
  sample = TFCSAnalyzerBase::GetInfo(sampleData.c_str(), dsid);

  std::string input = sample.inputSample;
  std::string baselabel = sample.label;
  int pdgid = sample.pdgid;
  int energy = sample.energy;
  float etamin = sample.etamin;
  float etamax = sample.etamax;
  int zv = sample.zv;

  std::cout << " *************************** " << std::endl;
  std::cout << " DSID : " << dsid << std::endl;
  std::cout << " location: " << input << std::endl;
  std::cout << " base name:  " << baselabel << std::endl;
  std::cout << " pdgID: " << pdgid << std::endl;
  std::cout << " energy (MeV) : " << energy << std::endl;
  std::cout << " eta main, max : " << etamin << " , " << etamax << std::endl;
  std::cout << " z vertex : " << zv << std::endl;
  std::cout << "*********************************" << std::endl;

  /////////////////////////////////////////
  // form names for ouput files and directories
  ///////////////////////////////////////////

  TString inputSample(Form("%s", input.c_str()));
  TString pcaSample(Form("%s%s.firstPCA.%s.root", topDir.c_str(),
                         baselabel.c_str(), version.c_str()));
  TString shapeSample(Form("%s%s.shapepara.%s.root", topDir.c_str(),
                           baselabel.c_str(), version.c_str()));
  TString plotDir(Form("%s/%s.plots.%s/", topPlotDir.c_str(), baselabel.c_str(),
                       version.c_str()));

  TString pcaAppSample = pcaSample;
  pcaAppSample.ReplaceAll("firstPCA", "firstPCA_App");

  /////////////////////////////////////////
  // read input sample and create first pca
  ///////////////////////////////////////////

  TChain* inputChain = new TChain("FCS_ParametrizationInput");
  inputChain->Add(inputSample);

  int nentries = inputChain->GetEntries();

  std::cout << " * Prepare to run on: " << inputSample
            << " with entries = " << nentries << std::endl;

  //--------- should be removed, use EnergyParam package -----
  // firstPCA *myfirstPCA = new firstPCA(inputChain, pcaSample.Data());
  // myfirstPCA->set_cumulativehistobins(5000);
  // myfirstPCA->set_edepositcut(0.001);
  // myfirstPCA->apply_etacut(0); //this flag is for the old files, which are
  // already sliced in eta
  // myfirstPCA->set_etacut(0.2, 0.25);
  // myfirstPCA->set_pcabinning(5, 1);
  // myfirstPCA->run();

  TFCSMakeFirstPCA* myfirstPCA =
      new TFCSMakeFirstPCA(inputChain, pcaSample.Data());
  myfirstPCA->set_cumulativehistobins(5000);
  myfirstPCA->set_edepositcut(0.001);
  myfirstPCA->apply_etacut(0);
  myfirstPCA->run();
  delete myfirstPCA;
  cout << "TFCSMakeFirstPCA done" << endl;

  int npca1 = 5;
  int npca2 = 1;

  TFCSApplyFirstPCA* myfirstPCA_App = new TFCSApplyFirstPCA(
      /*inputChain, pcaSample.Data(),*/ pcaAppSample.Data());
  myfirstPCA_App->set_pcabinning(npca1, npca2);
  myfirstPCA_App->set_edepositcut(myfirstPCA->get_edepositcut());
  myfirstPCA_App->run();
  delete myfirstPCA_App;
  cout << "TFCSApplyFirstPCA done" << endl;

  // delete inputChain;

  // -------------------------------------------------------

  TChain* pcaChain = new TChain("tree_1stPCA");
  pcaChain->Add(pcaAppSample);
  inputChain->AddFriend("tree_1stPCA");

  /////////////////////////////////////// ///
  // get relevant layers and no. of PCA bins
  // from the firstPCA
  ////////////////////////////////////////////

  TFile* fpca = TFile::Open(pcaAppSample);
  std::vector<int> v_layer;

  TH2I* relevantLayers = (TH2I*)fpca->Get("h_layer");
  int npca = relevantLayers->GetNbinsX();
  for (int ibiny = 1; ibiny <= relevantLayers->GetNbinsY(); ibiny++) {
    if (relevantLayers->GetBinContent(1, ibiny) == 1)
      v_layer.push_back(ibiny - 1);
  }

  std::cout << " relevantLayers = ";
  for (auto i : v_layer) std::cout << i << " ";
  std::cout << "\n";

  //////////////////////////////////////////////////////////
  ///// Create validation steering
  //////////////////////////////////////////////////////////

  TFile* f = new TFile(shapeSample, "recreate");

  for (int ilayer = 0; ilayer < v_layer.size(); ilayer++) {
    for (int ipca = 1; ipca <= npca; ipca++) {

      int analyze_layer = v_layer.at(ilayer);
      int analyze_pcabin = ipca;

      prefixlayer = Form("cs%d_", analyze_layer);
      prefixall = Form("cs%d_pca%d_", analyze_layer, analyze_pcabin);
      prefixEbin = Form("pca%d_", analyze_pcabin);

      TFCSShapeValidation analyze(inputChain, /*"",*/ analyze_layer);
      analyze.set_IsNewSample(true);
      analyze.set_Nentries(nentries);
      analyze.set_Debug(0);

      std::cout << "=============================" << std::endl;

      //////////////////////////////////////////////////////////
      ///// Chain to read 2D alpha_radius in mm from the input file
      //////////////////////////////////////////////////////////

      TFCSParametrizationChain RunInputHits(
          "input_EnergyAndHits", "original energy and hits from input file");

      TFCSValidationEnergyAndHits input_EnergyAndHits(
          "input_EnergyAndHits", "original energy and hits from input file",
          &analyze);

      input_EnergyAndHits.set_pdgid(pdgid);
      input_EnergyAndHits.set_calosample(analyze_layer);
      input_EnergyAndHits.set_Ekin_bin(analyze_pcabin);

      RunInputHits.push_back(&input_EnergyAndHits);
      RunInputHits.Print();

      TFCSValidationHitSpy hitspy_orig("hitspy_2D_E_alpha_radius",
                                       "shape parametrization");

      hitspy_orig.set_calosample(analyze_layer);

      int binwidth = 5;
      if (analyze_layer == 1 or analyze_layer == 5) binwidth = 1;
      float ymin = 0;
      float ymax = 10000;
      int nbinsy = (int)((ymax - ymin) / binwidth);

      TH2* h_orig_hitEnergy_alpha_r = new TH2F();

      if (isPhisymmetry) {
        h_orig_hitEnergy_alpha_r =
            analyze.InitTH2(prefixall + "hist_hitenergy_alpha_radius", "2D", 8,
                            0, TMath::Pi(), nbinsy, ymin, ymax);
        hitspy_orig.hist_hitenergy_alpha_absPhi_radius() =
            h_orig_hitEnergy_alpha_r;

      } else {
        h_orig_hitEnergy_alpha_r =
            analyze.InitTH2(prefixall + "hist_hitenergy_alpha_radius", "2D", 8,
                            0, 2 * TMath::Pi(), nbinsy, ymin, ymax);
        hitspy_orig.hist_hitenergy_alpha_radius() = h_orig_hitEnergy_alpha_r;
      }

      input_EnergyAndHits.push_back(&hitspy_orig);
      analyze.validations().emplace_back(&RunInputHits);

      std::cout << "=============================" << std::endl;
      //////////////////////////////////////////////////////////
      analyze.LoopEvents(analyze_pcabin);

      TH2F* h_alpha_r = new TH2F();

      h_alpha_r =
          (TH2F*)Create2DHistogram(h_orig_hitEnergy_alpha_r, energy_cutoff);
      // save polar plots
      CreatePolarPlot(h_alpha_r, plotDir.Data());

      f->cd();
      h_alpha_r->Write();

      if (h_orig_hitEnergy_alpha_r) delete h_orig_hitEnergy_alpha_r;
      if (h_alpha_r) delete h_alpha_r;
    }
  }

  f->Close();
}
