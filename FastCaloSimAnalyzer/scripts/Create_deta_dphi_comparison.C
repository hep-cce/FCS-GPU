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


void Create_deta_dphi_comparison(string particle, bool isEta) {

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

    TFile *out = new TFile("photon.root", "recreate");



    for (int ilayer = 0; ilayer < vlayer.size(); ilayer++) {

        int calolayer = vlayer.at(ilayer);

        for (int ipca = 0; ipca < vpca.size(); ipca++) {

            int PCAbin = vpca.at(ipca);

            string fileName = particle + "_" + std::to_string(energy) + "GeV" + "_eta_" + std::to_string(etamin) + "_" + std::to_string(etamax) + "_layer" + std::to_string(calolayer) + "_PCAbin" + std::to_string(PCAbin);


            string NewfileName = topDir + particle + "/HitsNewAlphaDr_" + fileName + ".root";
            string OldfileName = topDir + particle + "/HitsAlphaDr_" + fileName + ".root";


            TFile *Newfile = TFile::Open(NewfileName.c_str());
            TFile *Oldfile = TFile::Open(OldfileName.c_str());


            std::string histo = "";
            if (isEta) histo = "eta";
            else histo = "phi";

            string outfile = topDir + particle + "/comparison_plots/" + histo + "_" + std::to_string(calolayer) + "_" + std::to_string(PCAbin) + ".pdf";

            string label = "";
            string label2 = particle + " " + std::to_string(int(energy)) + " GeV " +  "0.2 < |#eta| < 0.25"  ;

            if (PCAbin == 99)
                label = "layer " + std::to_string(calolayer) + ", pca incl. ";
            else
                label = "layer " + std::to_string(calolayer) + ", pca " + std::to_string(PCAbin);

            string xlabel = "r*#delta#phi [mm]";
            if (isEta) xlabel = "r*#delta#eta [mm]";

            string ylabel = "a.u.";


            TH1F *h_eta_new_noExtr = (TH1F*)Newfile->Get(("hd" + histo + "E_entr_mm").c_str());
            TH1F *h_eta_new_Extr = (TH1F*)Newfile->Get(("hd" + histo + "E_mm").c_str());
            TH1F *h_eta_old_noExtr = (TH1F*)Oldfile->Get(("hd" + histo + "E_mm").c_str());

            // TH1F *h_eta_new_Extr = (TH1F*)Newfile->Get("hdphi");
            // TH1F *h_eta_old_noExtr = (TH1F*)Oldfile->Get("hdphi");


            h_eta_new_noExtr->SetLineColor(kBlue + 2);
            h_eta_new_noExtr->SetMarkerColor(kBlue + 2);

            h_eta_new_noExtr->SetFillColor(0);
            h_eta_new_noExtr->Scale(1 / h_eta_new_noExtr->Integral());
            h_eta_new_noExtr->SetAxisRange(-20, +20, "X");



            h_eta_old_noExtr->SetLineColor(kOrange + 4);
            h_eta_old_noExtr->SetMarkerColor(kOrange + 4);

            h_eta_old_noExtr->SetFillColor(0);
            h_eta_old_noExtr->Scale(1 / h_eta_old_noExtr->Integral());
            h_eta_old_noExtr->SetAxisRange(-20, +20, "X");

            h_eta_new_Extr->SetLineColor(kGreen + 3);
            h_eta_new_Extr->SetMarkerColor(kGreen + 3);

            h_eta_new_Extr->SetFillColor(0);
            h_eta_new_Extr->Scale(1 / h_eta_new_Extr->Integral());
            h_eta_new_Extr->SetAxisRange(-20, +20, "X");


            h_eta_new_Extr->SetMaximum(1.5 * h_eta_new_Extr->GetMaximum());


            h_eta_new_Extr->GetXaxis()->SetTitle(xlabel.c_str());
            h_eta_new_Extr->GetYaxis()->SetTitle(ylabel.c_str());


            TCanvas* c1 = new TCanvas("c1", "", 800, 600);
            TPad* thePad = (TPad*)c1->cd();


            h_eta_new_Extr->Draw("hist EX2 ");
            h_eta_new_noExtr->Draw("hist EX2 same");
            h_eta_old_noExtr->Draw("hist EX2 same");



            TLegend* leg = new TLegend(0.55, 0.8, 0.95, 0.9);
            leg->SetBorderSize(0);
            leg->SetFillStyle(0);
            leg->SetFillColor(0);
            leg->SetTextSize(0.04);
            leg->AddEntry(h_eta_old_noExtr, "2016 TTC entr.");
            leg->AddEntry(h_eta_new_noExtr, "2017 TTC entr.");
            leg->AddEntry(h_eta_new_Extr, "2017 TTC (entr.+back)/2");
            leg->Draw();



            myText(0.75,  0.75, 1, "#sqrt{s} = 13 TeV");
            ATLASLabel(0.18, 0.05, "Simulation Internal");
            myText(0.2, 0.9, 1, label2.c_str());
            myText(0.2, 0.85, 1, label.c_str());


            c1->SaveAs(outfile.c_str());

            out->cd();
            c1->Write();
            c1->Close();



        }
    }






}