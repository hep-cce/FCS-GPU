/*
  Copyright (C) 2002-2017 CERN for the benefit of the ATLAS collaboration
*/

#include "TFile.h"
#include "TH1D.h"
#include "TLegend.h"
#include "TKey.h"
#include "TString.h"
#include "TLine.h"
#include "TLatex.h"
#include "TCanvas.h"
#include "TStyle.h"
#include "TROOT.h"
#include "TLatex.h"


#include "atlasstyle/AtlasStyle.C"


#ifdef __CLING__
// these are not headers - do not treat them as such - needed for ROOT6
#include "atlasstyle/AtlasLabels.C"
#include "atlasstyle/AtlasUtils.C"
#endif


void PlotTTCetaphi(string particle, string histo, string xlabel) {

    gROOT->SetBatch(kTRUE);

#ifdef __CINT__
    gROOT->LoadMacro("atlasstyle/AtlasLabels.C");
    gROOT->LoadMacro("atlasstyle/AtlasUtils.C");
#endif

    SetAtlasStyle();


    float  energy   = 50;     // in GeV
    float  etamin   = 0.20;
    float  etamax   = 0.25;

    std::vector<int> vlayer;
    std::vector<int> vpca;

    vpca.push_back(99);
    // vpca.push_back(1);
    // vpca.push_back(2);
    // vpca.push_back(3);
    // vpca.push_back(4);
    // vpca.push_back(5);





    if (particle == "photon" or particle == "el_1mm") {
        vlayer.push_back(0);
        vlayer.push_back(1);
        vlayer.push_back(2);
        vlayer.push_back(3);
        vlayer.push_back(12);

    } else if (particle == "pionplus" or particle == "pionminus") {

        vlayer.push_back(0);
        vlayer.push_back(1);
        vlayer.push_back(2);
        vlayer.push_back(3);
        vlayer.push_back(12);
        vlayer.push_back(13);
        vlayer.push_back(14);

    } else {
        std::cerr << "Cant determine the particle type!" << std::endl ;
    }


    string topDir = "../../../run/output/shape_para/";

    system(("mkdir -p " + topDir + particle + "/comparison_plots").c_str());


    TCanvas* c1 = new TCanvas("c1", "", 800, 600);


    TLegend* leg = new TLegend(0.7, 0.7, 0.95, 0.9);
    leg->SetBorderSize(0);
    leg->SetFillStyle(0);
    leg->SetFillColor(0);
    leg->SetTextSize(0.04);


    string label = particle + " 50 GeV 0.20 < |#eta| < 0.25";


    for (int ilayer = 0; ilayer < vlayer.size(); ilayer++) {

        int calolayer = vlayer.at(ilayer);

        for (int ipca = 0; ipca < vpca.size(); ipca++) {

            int PCAbin = vpca.at(ipca);

            string fileName = particle + "_" + std::to_string(energy) + "GeV" + "_eta_" + std::to_string(etamin) + "_" + std::to_string(etamax) + "_layer" + std::to_string(calolayer) + "_PCAbin" + std::to_string(PCAbin);


            string NewfileName = topDir + particle + "/HitsNewAlphaDr_" + fileName + ".root";

            TFile *Newfile = TFile::Open(NewfileName.c_str());





            string ylabel = "a.u.";


            TH1F *hist = (TH1F*)Newfile->Get(histo.c_str());
            hist->GetXaxis()->SetTitle(xlabel.c_str());
            hist->GetYaxis()->SetTitle(ylabel.c_str());

            hist->Scale(1 / hist->Integral());
            hist->SetLineColor(ilayer + 2);
            hist->SetMarkerColor(ilayer + 2);

            hist->SetMaximum(2 * hist->GetMaximum());



            c1->cd();


            if (ilayer == 0 && ipca == 0 )
                hist->Draw("hist EX2");
            else
                hist->Draw("hist EX2 same");


            leg->AddEntry(hist, ("layer " + std::to_string(calolayer)).c_str());


        }
    }



    leg->Draw();

    // myText(0.7,  0.7, 1, "#sqrt{s} = 13 TeV");
    ATLASLabel(0.18, 0.05, "Simulation Internal");
    myText(0.18, 0.9, 1, label.c_str());


    string outfile = topDir + particle + "/comparison_plots/" + histo + ".pdf";

    c1->SaveAs(outfile.c_str());
    c1->Close();






}