/*
  Copyright (C) 2002-2018 CERN for the benefit of the ATLAS collaboration
*/

// #include "TFCSfirstPCA.h"
// #include "TFCSAnalyzerBase.h"
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

#include "../atlasstyle/AtlasStyle.C"


#ifdef __CLING__
// these are not headers - do not treat them as such - needed for ROOT6
#include "../atlasstyle/AtlasLabels.C"
#include "../atlasstyle/AtlasUtils.C"
#endif


TH1F* GetEfficiencyHistogram(std::string fileName, int layer, bool above = false, std::string label = "") {


    TFile *f = TFile::Open(fileName.c_str());
    cout << f << endl;




    TString name = Form("cs%i_efficiency%s", layer, label.c_str());
    if (above) name = Form("cs%i_efficiency_above_etaboundary%s", layer, label.c_str());




    TCanvas* c = (TCanvas*)f->Get(Form("cs%i_wiggle_correction", layer));
    if (above) c = (TCanvas*)f->Get(Form("cs%i_above_etaboundary_wiggle_correction", layer));
    cout << c << endl;

    TPad* p = (TPad*)c->GetPrimitive(Form("cs%i_wiggle_correction_3", layer));
    if (above) p = (TPad*)c->GetPrimitive(Form("cs%i_above_etaboundary_wiggle_correction_3", layer));
    cout << p << endl;

    TH1F* h = (TH1F*)p->GetPrimitive(Form("cs%i_efficiency", layer));
    if (above) h = (TH1F*)p->GetPrimitive(Form("cs%i_above_etaboundary_efficiency", layer));

    cout << h->GetName() << endl;



    TH1F* h_eff = (TH1F*)h->Clone();

    cout << h_eff->GetNbinsX() << endl;




    h_eff->SetNameTitle(name, name);

    // f->Close();

    return h_eff;
}

TCanvas* GetCanvas(int layer, bool above = false, std::string topDir = "../macro/wiggle_output/") {

    cout << " ===> looking at layer: " << layer << " above = " << above << endl;
    gStyle->SetOptStat(0);
    std::vector<std::string> fileNameVector;
    std::vector<std::string> legVector;

    if (abs(layer) >= 0 and abs(layer) < 4) {
        fileNameVector.push_back("mc16_13TeV.430610.ParticleGun_pid22_E4096_disj_eta_m55_m50_50_55_zv_0.wiggleDerivative.ver01.root");
        fileNameVector.push_back("mc16_13TeV.431001.ParticleGun_pid22_E65536_disj_eta_m10_m5_5_10_zv_0.wiggleDerivative.ver01.root");
        fileNameVector.push_back("mc16_13TeV.431004.ParticleGun_pid22_E65536_disj_eta_m25_m20_20_25_zv_0.wiggleDerivative.ver01.root");
        if (layer != 0) fileNameVector.push_back("mc16_13TeV.431609.ParticleGun_pid22_E4194304_disj_eta_m50_m45_45_50_zv_0.wiggleDerivative.ver01.root");


        legVector.push_back("E4096_eta_50_55");
        legVector.push_back("E65536_eta_5_10");
        legVector.push_back("E65536_eta_20_25");
        if (layer != 0)legVector.push_back("E4194304_eta_45_50");
    } else {

        fileNameVector.push_back("mc16_13TeV.430735.ParticleGun_pid22_E8192_disj_eta_m180_m175_175_180_zv_0.wiggleDerivative.ver01.root");
        if (abs(layer) == 6 or layer == 7) fileNameVector.push_back("mc16_13TeV.431056.ParticleGun_pid22_E65536_disj_eta_m285_m280_280_285_zv_0.wiggleDerivative.ver01.root");
        if (abs(layer) == 6 or layer == 7)  fileNameVector.push_back("mc16_13TeV.431361.ParticleGun_pid22_E524288_disj_eta_m310_m305_305_310_zv_0.wiggleDerivative.ver01.root");
        if (abs(layer) == 6 or layer == 7) fileNameVector.push_back("mc16_13TeV.431165.ParticleGun_pid22_E131072_disj_eta_m330_m325_325_330_zv_0.wiggleDerivative.ver01.root");
        fileNameVector.push_back("mc16_13TeV.431633.ParticleGun_pid22_E4194304_disj_eta_m170_m165_165_170_zv_0.wiggleDerivative.ver01.root");

        legVector.push_back("E8192_eta_175_180");
        if (abs(layer) == 6 or layer == 7) legVector.push_back("E65536_eta_280_285");
        if (abs(layer) == 6 or layer == 7) legVector.push_back("E524288_eta_305_310");
        if (abs(layer) == 6 or layer == 7) legVector.push_back("E131072_eta_325_330");
        legVector.push_back("E4194304_eta_165_170");

    }


    std::vector<int> color{1, 2, 3, 4, 6, 9};


    std::string postfix = "";
    if (above) postfix = "_above_etaboundary";

    TCanvas * c1 = new TCanvas(Form("cs%i_efficiency%s", layer, postfix.c_str()), Form("cs%i_efficiency%s", layer, postfix.c_str()), 0, 0, 1000, 800);

    TLegend* l = new TLegend(0.2 , 0.2, 0.35, 0.35);

    for (int i = 0; i <  fileNameVector.size(); i++) {
        std::string file = topDir + fileNameVector.at(i);
        std::string leg = legVector.at(i);
        cout << file << endl;
        TH1F* h1 = GetEfficiencyHistogram(file, layer, above);
        cout << h1->GetName() << endl;
        h1->SetLineColor(color.at(i));
        h1->SetMarkerColor(color.at(i));
        h1->SetStats(0);
        l->AddEntry(h1, leg.c_str(), "lp");
        c1->cd();
        if (i == 0) h1->Draw();
        else
            h1->Draw("same");
    }
    l->Draw();

    return c1;
}

void runWiggleEfficiencyComparisonPlots(std::string topDir = "../macro/wiggle_output/")
{



    std::vector<int> layers{0, 1, -1, 2, 3, 4, 5, 6, -6, 7};



    for (int i = 0 ; i < layers.size(); i++) {
        int layer = layers.at(i);
        bool above = false;
        if (layer == -1 or layer == -6) above = true;
        TCanvas* c1 = GetCanvas(abs(layer), above);

        if (i == 0) c1->Print("wiggle_validation.pdf(", "pdf");
        else if (i == layers.size() - 1) c1->Print("wiggle_validation.pdf)", "pdf");
        else c1->Print("wiggle_validation.pdf", "pdf");
    }



}