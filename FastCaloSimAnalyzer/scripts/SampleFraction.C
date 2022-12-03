/*
  Copyright (C) 2002-2023 CERN for the benefit of the ATLAS collaboration
*/

#include "TROOT.h"
#include "TTree.h"
#include "TFile.h"
#include "TCanvas.h"
#include "TH1.h"
#include "TString.h"

#include "../atlasstyle/AtlasStyle.C"


#ifdef __CLING__
// these are not headers - do not treat them as such - needed for ROOT6
#include "../atlasstyle/AtlasLabels.C"
#include "../atlasstyle/AtlasUtils.C"
#endif


using namespace std;


void SampleFraction();

void SampleFraction() {

  gROOT->SetBatch(kTRUE);

#ifdef __CINT__
  gROOT->LoadMacro("../atlasstyle/AtlasLabels.C");
  gROOT->LoadMacro("../atlasstyle/AtlasUtils.C");
#endif

  SetAtlasStyle();



  int GeV = 1e3;

  std::string topDir = "/eos/atlas/atlascerngroupdisk/proj-simul/InputSamplesLocalProd2017/rel_21_0_41/Samples/";
  topDir = "./";

  std::vector<string> v_eta;

  v_eta.push_back("eta000_005");
  v_eta.push_back("eta100_105");
  v_eta.push_back("eta140_145");
  v_eta.push_back("eta200_205");
  v_eta.push_back("eta300_305");
  v_eta.push_back("eta400_405");


  system("mkdir -p plots_sampfrac");

  TFile* fout = new TFile("plots_sampfrac/sampfrac.root", "RECREATE");

  for (unsigned int ieta = 0; ieta < v_eta.size(); ieta++) {

    std::vector<bool> *is_emec = 0;
    std::vector<float> *hit_energy = 0;
    std::vector<float> *g4_energy = 0;
    std::vector<float> *hit_sampling = 0;
    std::vector<float> *g4_sampling = 0;
    std::vector<float> *hitsampfrac = 0;
    std::vector<float> *g4sampfrac = 0;

    std::string eta = v_eta.at(ieta);
    std::string filename = "ISF_HitAnalysis." + eta + ".root";
    cout << " filename = " << filename << endl;

    TFile *f = TFile::Open(filename.c_str());

    TTree *t = (TTree*)f->Get("ISF_HitAnalysis/CaloHitAna");

    t->SetBranchAddress("HitIsLArEndCap", &is_emec);
    t->SetBranchAddress("HitE", &hit_energy);
    t->SetBranchAddress("G4HitE", &g4_energy);

    t->SetBranchAddress("HitSampling", &hit_sampling);
    t->SetBranchAddress("G4HitSampling", &g4_sampling);


    t->SetBranchAddress("HitSamplingFraction", &hitsampfrac);
    t->SetBranchAddress("G4HitSamplingFraction", &g4sampfrac);



    TH1F *h_hit = new TH1F(Form("sampling_hit_%s", eta.c_str()), Form("sampling_hit_%s", eta.c_str()), 25, 0, 25);
    TH1F *h_g4 = new TH1F(Form("sampling_g4_%s", eta.c_str()), Form("sampling_g4_%s", eta.c_str()), 25, 0, 25);


    TH1F *h2_hit = new TH1F(Form("sampling2_hit_%s", eta.c_str()), Form("sampling2_hit_%s", eta.c_str()), 25, 0, 25);
    TH1F *h2_g4 = new TH1F(Form("sampling2_g4_%s", eta.c_str()), Form("sampling2_g4_%s", eta.c_str()), 25, 0, 25);

    h_hit->Sumw2();
    h_g4->Sumw2();
    h2_hit->Sumw2();
    h2_g4->Sumw2();



    h_hit->SetTitle(eta.c_str());


    h_hit->GetXaxis()->SetTitle("layer");
    h_g4->GetXaxis()->SetTitle("layer");

    h_hit->GetYaxis()->SetTitle("E [GeV]");
    h_g4->GetYaxis()->SetTitle("E [GeV]");

    h2_hit->GetXaxis()->SetTitle("layer");
    h2_g4->GetXaxis()->SetTitle("layer");

    h2_hit->GetYaxis()->SetTitle("E [GeV]");
    h2_g4->GetYaxis()->SetTitle("E [GeV]");


    h_hit->SetLineColor(kRed);
    h_g4->SetLineColor(kBlue);
    h_hit->SetMarkerColor(kRed);
    h_g4->SetMarkerColor(kBlue);

    h2_hit->SetLineColor(6);
    h2_g4->SetLineColor(7);
    h2_hit->SetMarkerColor(6);
    h2_g4->SetMarkerColor(7);


    double nentries = t->GetEntries();

    cout << " Entries = " << nentries << endl;

    for (int ievent = 0; ievent < nentries; ievent++)
    {

      t->GetEntry(ievent);

      for (unsigned int x = 0; x < hitsampfrac->size(); x++)
      {
        float layer = hit_sampling->at(x);
        if (layer >= 12 and layer <= 20) {
          h_hit->Fill(layer, hitsampfrac->at(x) * (hit_energy->at(x) / GeV));
          h2_hit->Fill(layer, hit_energy->at(x) / GeV);

        } else {
          h_hit->Fill(layer, (hit_energy->at(x) / GeV) / hitsampfrac->at(x)  );
          h2_hit->Fill(layer, hit_energy->at(x) / GeV );
        }
      }

      for (unsigned int y = 0; y < g4sampfrac->size(); y++)
      {
        float layer = hit_sampling->at(y);
        if (layer >= 12 and layer <= 20 ) {
          h_g4->Fill(layer, g4sampfrac->at(y) * (g4_energy->at(y) / GeV));
          h2_g4->Fill(layer, g4_energy->at(y) / GeV);

        } else {
          h_g4->Fill(layer,  (g4_energy->at(y) / GeV) / g4sampfrac->at(y));
          h2_g4->Fill(layer,  g4_energy->at(y) / GeV);
        }
      }

    } // end loop over events

    nentries = 1 / nentries;
    cout << " nentries = " << nentries << endl;
    h_hit->Scale(nentries);
    h_g4->Scale(nentries);
    h2_hit->Scale(nentries);
    h2_g4->Scale(nentries);



    float hit_integral_weight =  h_hit->Integral() ;
    float g4_integral_weight  = h_g4->Integral() ;


    float hit_integral_noweight =  h2_hit->Integral() ;
    float g4_integral_noweight  = h2_g4->Integral() ;


    h_g4->SetMaximum(1.5 * h_g4->GetMaximum());



    fout->cd();

    TCanvas* c1 = new TCanvas(Form("c%s", eta.c_str()), Form("c%s", eta.c_str()), 800, 600);
    TPad* thePad = (TPad*)c1->cd();

    thePad->cd();
    // thePad->SetLogy();
    h_g4->Draw("hist EX2");
    h_hit->Draw("hist EX2 same");
    h2_g4->Draw("hist EX2 same");
    h2_hit->Draw("hist EX2 same");


    TLegend* leg = new TLegend(0.55, 0.8, 0.95, 0.9);
    leg->SetBorderSize(0);
    leg->SetFillStyle(0);
    leg->SetFillColor(0);
    leg->SetTextSize(0.02);
    leg->AddEntry(h_hit, Form("calo weighted (Etot: %.2f)", hit_integral_weight));
    leg->AddEntry(h_g4, Form("g4 weighted (Etot: %.2f)", g4_integral_weight));
    leg->AddEntry(h2_hit, Form("calo raw (Etot: %.2f)", hit_integral_noweight));
    leg->AddEntry(h2_g4, Form("g4 raw (Etot: %.2f)", g4_integral_noweight));
    leg->Draw();


    ATLASLabel(0.18, 0.05, "Simulation Internal");
    myText(0.2, 0.9, 1, ("photon_50GeV_" + eta).c_str());

    std::string pdf = "plots_sampfrac/" + eta + ".pdf";



    c1->SaveAs(pdf.c_str());
    h_hit->Write();
    h_g4->Write();
    h2_hit->Write();
    h2_g4->Write();
    f->Close();
    // fout->Close();
  } // end loop over etas

  fout->Close();
}







