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


void PlotEtaPhi2D(string particle, string histo) {

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



            TFile *Newfile = TFile::Open(NewfileName.c_str());





            string outfile = topDir + particle + "/comparison_plots/" + histo + "_" + std::to_string(calolayer) + "_" + std::to_string(PCAbin) + ".pdf";

            string label = "";
            string label2 = particle + " " + std::to_string(int(energy)) + " GeV " +  "0.2 < |#eta| < 0.25"  ;

            if (PCAbin == 99)
                label = "layer " + std::to_string(calolayer) + ", pca incl. ";
            else
                label = "layer " + std::to_string(calolayer) + ", pca " + std::to_string(PCAbin);

            string xlabel =  "r*#delta#eta [mm]";
            string ylabel = "r*#delta#phi [mm]";
            string zlabel = "Energy normalized to 1";

            TH2F * hist = (TH2F*)Newfile->Get(histo.c_str());
            hist->Scale(1 / hist->Integral());

            hist->GetXaxis()->SetTitle(xlabel.c_str());
            hist->GetYaxis()->SetTitle(ylabel.c_str());
            hist->GetYaxis()->SetTitleOffset(0.8);
            hist->GetZaxis()->SetTitle(zlabel.c_str());
            hist->GetZaxis()->SetTitleOffset(0.9);
            hist->GetZaxis()->SetLabelSize(0.035);




            hist->SetAxisRange(-6, +6, "X");
            hist->SetAxisRange(-6, +6, "Y");



            TCanvas* c1 = new TCanvas("c1", "", 900, 600);
            TPad* thePad = (TPad*)c1->cd();
            thePad->SetGridx();
            thePad->SetGridy();
            thePad->SetLogz();
            thePad->SetLeftMargin(-3.5);
            thePad->SetRightMargin(3.5);
            gStyle->SetPalette(kRainBow);


            hist->Draw("colz");


            gPad->Update();
            TPaletteAxis *palette = (TPaletteAxis*)hist->GetListOfFunctions()->FindObject("palette");
            //palette->GetAxis()->SetLabelSize(0.001);
            palette->SetX1NDC(0.85);
            palette->SetX2NDC(0.9);
            gPad->Modified();
            gPad->Update();





            myText(0.7,  0.96, 1, "#sqrt{s} = 13 TeV");
            ATLASLabel(0.1, 0.05, "Simulation Internal");
            myText(0.1, 0.96, 1, label2.c_str());
            myText(0.5, 0.96, 1, label.c_str());


            c1->SaveAs(outfile.c_str());

            out->cd();
            c1->Write();
            c1->Close();



        }
    }






}