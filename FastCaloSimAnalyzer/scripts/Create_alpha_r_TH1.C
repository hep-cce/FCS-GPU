/*
  Copyright (C) 2002-2017 CERN for the benefit of the ATLAS collaboration
*/

#include "TROOT.h"
#include "TFile.h"
#include "TH1.h"
#include "TH2.h"
#include "TString.h"
#include "TLegend.h"

#include <iostream>
#include <string>

void Create_alpha_r_TH1(string particle, bool isNewSample) {

    gROOT->SetBatch(kTRUE);
    gStyle->SetOptStat(0);


    string topDir = "/Users/ahasib/Documents/Analysis/Projects/Athena/FCS_ShapeParametrization_dir/Simulation/ISF/ISF_FastCaloSim/run/output/shape_para/";


    string filename = "";

    if (isNewSample)
        filename = particle + "/HitsNewAlphaDr_" + particle + "_50.000000GeV_eta_0.200000_0.250000_layer2_PCAbin1.root";
    else
        filename = particle + "/HitsAlphaDr_" + particle + "_50.000000GeV_eta_0.200000_0.250000_layer2_PCAbin1.root";


    string outfile = "";

    if (isNewSample)
        outfile = particle + "New_alpha_r.root";
    else
        outfile = particle + "_alpha_r.root";

    string file = topDir + filename;

    std::cout << "file = " << file.c_str() << std::endl ;

    TFile *f = TFile::Open(file.c_str());

    string prefix = "";
    if (particle == "pion") prefix =  "#pi^{-} :  ";
    else if (particle == "pionplus") prefix =  "#pi^{+} :  ";
    else if (particle == "photon") prefix = "#gamma^{0}  :";
    else prefix = "none";

    f->cd();

    TH2F* hist = (TH2F*)f->Get("h_alphaE_drE_rebin");

    TH1F* halpha = (TH1F*)f->Get("h_alphaE_rebin");
    halpha->Scale(1 / halpha->Integral());
    string alpha_title = prefix + " inclusive r ";
    halpha->SetTitle(alpha_title.c_str());
    halpha->GetXaxis()->SetTitle("#alpha ( 0 < #alpha < 2#pi )");
    halpha->GetYaxis()->SetTitle("a.u.");

    int nbinsy = hist->GetNbinsY();
    int nbinsx = hist->GetNbinsX();

    std::cout << "NbinsX, NbinsY = " << nbinsx << " , " <<  nbinsy << std::endl ;

    TFile *fout = new TFile(outfile.c_str(), "recreate");


    for (int i = 1; i <= nbinsy; ++i)
    {
        TH1F * h1 = new TH1F(Form("h_alpha_r%i", i), Form("h_alpha_r%i", i), nbinsx, TMath::Pi() / 8 , 2 * TMath::Pi() + TMath::Pi() / 8);




        string title = prefix + std::to_string(i - 1) + " <  r (mm) < " + std::to_string(i);

        h1->Sumw2();
        h1->SetTitle(title.c_str());
        h1->GetXaxis()->SetTitle("#alpha ( 0 < #alpha < 2#pi )");
        h1->GetYaxis()->SetTitle("a.u.");


        TCanvas *c1 = new TCanvas(Form("alpha_r%i_mm", i), Form("alpha_r%i_mm", i), 800, 600);


        for (int j = 0; j < nbinsx; ++j)
        {
            h1->SetBinContent(j, hist->GetBinContent(j, i));
            h1->SetBinError(j, hist->GetBinError(j, i));


        }



        h1->Scale(1 / h1->Integral());

        c1->cd();
        h1->Draw("hist same E1");
        // h1->Draw("PE same");
        fout->cd();
        // h1->Write();
        c1->Write();
    }

    TCanvas *c2 = new TCanvas("alpha", "alpha", 800, 500);
    c2->cd();
    halpha->Draw("hist same E1");
    c2->Write();


    fout->Close();

}