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




string prefixEbin;
string prefixEta;



void GetTH1TTreeDraw(TH1F*& hist, TTree* tree, std::string var, std::string cut = "")
{


    TH1F* histo = new TH1F();
    TString varexp(Form("%s>>histo", var.c_str()));

    // std::cout << "varexp = " << varexp << std::endl ;
    TString selection(Form("%s", cut.c_str()));
    // std::cout << "selection = " << selection << std::endl ;
    tree->Draw(varexp, selection, "goff");

    TH1F* htemp = (TH1F*)gROOT->FindObject("histo");
    hist = (TH1F*)htemp->Clone("hist");


    delete htemp;
    delete histo;

}


void GetTH1TTreeDraw(TH1F*& hist, TTree* tree, std::string var, std::string* cut, int nbins, double xmin, double xmax)
{


    TH1F* histo = new TH1F();
    TString varexp(Form("%s>>histo(%i, %f, %f)", var.c_str(), nbins, xmin, xmax));

    // std::cout << "varexp = " << varexp << std::endl ;
    TString selection(Form("%s", cut->c_str()));
    // std::cout << "selection = " << selection << std::endl ;
    tree->Draw(varexp, selection, "goff");

    TH1F* htemp = (TH1F*)gROOT->FindObject("histo");
    hist = (TH1F*)htemp->Clone("hist");


    delete htemp;
    delete histo;

}


TCanvas* PlotPolarHalf(TH2F* h, std::string label, std::string xlabel, std::string ylabel, std::string zlabel, int zoom_level)
{

    gStyle->SetPalette(kRainBow);
    // gStyle->SetOptStat(0);


    h->Sumw2();

    int nzoom = h->GetNbinsY() / zoom_level;
    float zoom = h->GetYaxis()->GetBinUpEdge(nzoom);


    h->GetYaxis()->SetRangeUser(0, float(zoom));
    h->GetYaxis()->SetLabelSize(.025);
    h->GetXaxis()->SetLabelSize(.025);
    h->GetXaxis()->SetTitle(xlabel.c_str());
    h->GetXaxis()->SetTitleSize(0.035);
    h->GetYaxis()->SetTitle(ylabel.c_str());
    h->GetYaxis()->SetTitleSize(0.035);

    // h->GetZaxis()->SetLabelSize(0.025);
    h->GetZaxis()->SetTitle(zlabel.c_str());
    // h->GetZaxis()->SetTitleSize(0.035);
    h->GetZaxis()->SetTitleOffset(1.4);


    TLatex* title = new TLatex(-zoom, 1.02 * zoom, label.c_str());
    // title->SetTextSize(0.03);
    // title->SetTextFont(42);

    TLatex* l = new TLatex(-1 * zoom, -1.30 * zoom, "ATLAS");
    // l->SetTextSize(.035);
    l->SetTextFont(72);

    TLatex* l2 = new TLatex(-0.5 * zoom, -1.30 * zoom, "Simulation Internal");
    // TLatex* l2 = new TLatex(-0.6 * zoom, -1.20 * zoom, "Simulation Preliminary");

    // l2->SetTextSize(.035);
    l2->SetTextFont(42);



    TCanvas* c1 = new TCanvas("c1", "", 1030, 900);
    c1->cd();
    c1->SetLeftMargin(0.14);
    c1->SetRightMargin(0.22);


    std::string frameTitle = "; " + xlabel + "; " + ylabel;
    gPad->DrawFrame(-zoom, -zoom, zoom, 0.5 * zoom , frameTitle.c_str());
    h->Draw("same colz pol");
    l->Draw();
    l2->Draw();
    title->Draw();
    c1->SetLogz();

    gPad->Update();
    TPaletteAxis *palette = (TPaletteAxis*)h->GetListOfFunctions()->FindObject("palette");
    palette->SetX1NDC(0.8);
    palette->SetX2NDC(0.84);
    palette->SetY1NDC(0.15);
    palette->SetY2NDC(0.95);
    gPad->Modified();
    gPad->Update();

    return c1;
}

TCanvas* PlotPolar(TH2F* h, std::string label, std::string xlabel, std::string ylabel, std::string zlabel, int zoom_level)
{

    gStyle->SetPalette(kRainBow);
    // gStyle->SetOptStat(0);


    h->Sumw2();

    int nzoom = h->GetNbinsY() / zoom_level;
    float zoom = h->GetYaxis()->GetBinUpEdge(nzoom);


    h->GetYaxis()->SetRangeUser(-float(zoom), float(zoom));
    h->GetYaxis()->SetLabelSize(.025);
    h->GetXaxis()->SetLabelSize(.025);
    h->GetXaxis()->SetTitle(xlabel.c_str());
    h->GetXaxis()->SetTitleSize(0.035);
    h->GetYaxis()->SetTitle(ylabel.c_str());
    h->GetYaxis()->SetTitleSize(0.035);

    // h->GetZaxis()->SetLabelSize(0.025);
    h->GetZaxis()->SetTitle(zlabel.c_str());
    // h->GetZaxis()->SetTitleSize(0.035);
    h->GetZaxis()->SetTitleOffset(1.4);


    TLatex* title = new TLatex(-zoom, 1.02 * zoom, label.c_str());
    // title->SetTextSize(0.03);
    // title->SetTextFont(42);

    TLatex* l = new TLatex(-1 * zoom, -1.30 * zoom, "ATLAS");
    // l->SetTextSize(.035);
    l->SetTextFont(72);

    TLatex* l2 = new TLatex(-0.5 * zoom, -1.30 * zoom, "Simulation Internal");
    // TLatex* l2 = new TLatex(-0.6 * zoom, -1.20 * zoom, "Simulation Preliminary");

    // l2->SetTextSize(.035);
    l2->SetTextFont(42);



    TCanvas* c1 = new TCanvas("c1", "", 1030, 900);
    c1->cd();
    c1->SetLeftMargin(0.14);
    c1->SetRightMargin(0.22);


    std::string frameTitle = "; " + xlabel + "; " + ylabel;
    gPad->DrawFrame(-zoom, -zoom, zoom, zoom , frameTitle.c_str());
    h->Draw("same colz pol");
    l->Draw();
    l2->Draw();
    title->Draw();
    c1->SetLogz();

    gPad->Update();
    TPaletteAxis *palette = (TPaletteAxis*)h->GetListOfFunctions()->FindObject("palette");
    palette->SetX1NDC(0.8);
    palette->SetX2NDC(0.84);
    palette->SetY1NDC(0.15);
    palette->SetY2NDC(0.95);
    gPad->Modified();
    gPad->Update();

    return c1;
}
void shapePlot() {

    TFile* fphotonFull = TFile::Open("/eos/atlas/atlascerngroupdisk/proj-simul/flatTTree/mc16_13TeV.431211.ParticleGun_pid22_E262144_disj_eta_m60_m55_55_60_zv_0.shapeparaFull.ver01.root");

    TH2F* h_full_photon = (TH2F*)fphotonFull->Get("cs2_pca5_hist_hitenergy_alpha_radius_2D");
    // TCanvas* cfullPhoton = PlotPolar(h_full_photon, "#gamma, 265GeV, 0.55 < |#eta| < 0.60, EM Barrel 2", "x [mm]", "y[mm]", "Normalized to unity", 4);

    // cfullPhoton->Print("photon_2D_full_shape_E265_eta55.pdf");


    TFile* fpionFull = TFile::Open("/eos/atlas/atlascerngroupdisk/proj-simul/flatTTree/mc16_13TeV.434611.ParticleGun_pid211_E262144_disj_eta_m60_m55_55_60_zv_0.shapeparaFull.ver01.root");

    TH2F* h_full_pion = (TH2F*)fpionFull->Get("cs13_pca5_hist_hitenergy_alpha_radius_2D");
    // TCanvas* cfullPion = PlotPolar(h_full_pion, "#pi^{#pm}, 265GeV, 0.55 < |#eta| < 0.60, Tile Barrel 2", "x [mm]", "y[mm]", "Normalized to unity", 8);

    // cfullPion->Print("pion_2D_full_shape_E265_eta55.pdf");


    TFile* fphotonHalf = TFile::Open("/eos/atlas/atlascerngroupdisk/proj-simul/flatTTree/mc16_13TeV.431004.ParticleGun_pid22_E65536_disj_eta_m25_m20_20_25_zv_0.shapepara.ver02.root");

    TH2F* h_half_photon = (TH2F*)fphotonHalf->Get("cs2_pca5_hist_hitenergy_alpha_radius_2D");
    TCanvas* cHalfPhoton = PlotPolar(h_half_photon, "#gamma, 65GeV, 0.20 < |#eta| < 0.25, EM Barrel 2", "x [mm]", "y[mm]", "Normalized to unity", 4);

    cHalfPhoton->Print("photon_2D_half_shape_E265_eta20.pdf");


}


TCanvas* plotComparison(TH1F* h1, TH1F* h2, bool isLog, bool isNumber, std::string xlabel = "", std::string ylabel = "", std::string label = "", std::string canvas = "")
{

    TGaxis::SetMaxDigits(4);


    h1->GetXaxis()->SetTitle(xlabel.c_str());
    h1->GetYaxis()->SetTitle(ylabel.c_str());

    h1->SetLineColor(kRed);
    h1->SetMarkerColor(kRed);
    h1->SetMarkerSize(1.5);



    h2->SetLineColor(kBlack);
    h2->SetLineStyle(7);

    h2->SetMarkerColor(kBlack);
    h2->SetMarkerStyle(kOpenCircle);
    h2->SetMarkerSize(1.5);

    float ymax = h1->GetMaximum();
    if (ymax < h2->GetMaximum())
        ymax = h2->GetMaximum();

    h1->SetMaximum(1.5 * ymax);
    if (isLog)
        h1->SetMaximum(50 * ymax);



    if (isLog) canvas = canvas + "_log";

    if (isNumber) {
        int div = h1->GetNbinsX();
        h1->SetNdivisions(div);
        h2->SetNdivisions(div);
    }


    TCanvas * c1 = new TCanvas(canvas.c_str(), canvas.c_str(), 1200, 900);
    // c1->SetLeftMargin(0.14);
    c1->SetRightMargin(0.09);

    c1->cd();

    if (isLog) c1->SetLogy();

    h1->Draw("hist E");
    h2->Draw("hist E same");

    TLegend* leg = new TLegend(0.7, 0.75, 0.9, 0.9);
    leg->SetBorderSize(0);
    leg->SetFillStyle(0);
    leg->SetFillColor(0);
    // leg->SetTextSize(0.02);
    leg->AddEntry(h2, "Geant4", "lpe");
    leg->AddEntry(h1, "FCS V2", "lpe");

    leg->Draw();




    ATLASLabel(0.2, 0.86, "Simulation Internal");
    myText(0.2, 0.76, 1, label.c_str());

    gPad->Update();



    return c1;


}

void plotTH1(TFile* fFCSV2, TFile* fG4, std::string tree, std::string var, int nbins, float xmin, float xmax, std::string cut = "", std::string xlabel = "", std::string ylabel = "", std::string label = "", bool isLog = false, bool isNumber = false)
{



    TTree* tFCSV2 = (TTree*)fFCSV2->Get(tree.c_str());
    TTree* tG4 = (TTree*)fG4->Get(tree.c_str());


    TH1F* hFCSV2temp = new TH1F();
    TH1F* hG4temp = new TH1F();

    TH1F* hFCSV2 = new TH1F();
    TH1F* hG4 = new TH1F();

    GetTH1TTreeDraw(hFCSV2temp, tFCSV2, var);
    GetTH1TTreeDraw(hG4temp, tG4, var);


    float integralFCSV2 = hFCSV2temp->Integral(0, -1);
    float integralG4 = hG4temp->Integral(0, -1);

    cout << " integral FCSV2 = " << integralFCSV2 << "integral G4 = " << integralG4 << endl;

    // xmin = hFCSV2temp->GetXaxis()->GetXmin();
    // xmax = hFCSV2temp->GetXaxis()->GetXmax();

    GetTH1TTreeDraw(hFCSV2, tFCSV2, var, &cut, nbins, xmin, xmax);
    GetTH1TTreeDraw(hG4, tG4, var, &cut, nbins, xmin, xmax);

    hFCSV2->Sumw2();
    hG4->Sumw2();

    hFCSV2->Scale(1 / integralFCSV2);
    hG4->Scale(1 / integralG4);

    string title = hFCSV2->GetTitle();
    // TString fileName = Form("%s_%s_%s", title.c_str(), prefixEbin.c_str(), prefixEta.c_str());
    string fileName = title + "_" + prefixEbin + "_" + prefixEta;
    if (isLog) fileName = fileName + "_log";

    TCanvas* c1 = plotComparison(hFCSV2, hG4, isLog, isNumber, xlabel, ylabel, label, fileName);
    c1->Print(("G4comparison_plot/" + fileName + ".pdf").c_str());



}


TCanvas* plotTProfile2D(TProfile2D* h, string xlabel = "", string ylabel = "", string zlabel = "", string label = "", string canvas = "") {

    gStyle->SetPalette(kRainBow);

    h->Sumw2();
    h->GetXaxis()->SetNdivisions(5);
    h->GetYaxis()->SetNdivisions(5);


    h->GetXaxis()->SetTitle(xlabel.c_str());
    h->GetYaxis()->SetTitle(ylabel.c_str());
    h->GetZaxis()->SetTitle(zlabel.c_str());



    TCanvas* c1 = new TCanvas(canvas.c_str(), canvas.c_str(), 1200, 900);
    c1->cd();
    // c1->SetLeftMargin(-0.24);
    c1->SetRightMargin(0.17);


    std::string frameTitle = "; " + xlabel + "; " + ylabel;

    h->Draw("same colz");


    // h->GetXaxis()->SetRangeUser(-0.12, 0.12);
    // h->GetYaxis()->SetRangeUser(-0.12, 0.12);
    // h->GetZaxis()->SetRangeUser(0.5, 2.0);


    ATLASLabel(0.18, 0.07, "Simulation Internal");
    myText(0.18, 0.96, 1, label.c_str());



    gPad->Update();
    TPaletteAxis *palette = (TPaletteAxis*)h->GetListOfFunctions()->FindObject("palette");
    palette->SetX1NDC(0.84);
    palette->SetX2NDC(0.89);
    // palette->SetY1NDC(0.2);
    // palette->SetY2NDC(0.8);
    gPad->Modified();
    gPad->Update();

    return c1;


}

void WiggleValidationPlot()
{

    TFile *fwiggle = TFile::Open("/eos/atlas/atlascerngroupdisk/proj-simul/flatTTree/2D_closure_plotsSampling_2.root");

    TProfile2D* hpre = (TProfile2D*)fwiggle->Get("dEta_dPhi_FCS_orig_overAvg_Layer_Sampling_2_Layer_0");
    TProfile2D* hpost = (TProfile2D*)fwiggle->Get("dEta_dPhi_FCS_matched_overAvg_Layer_Sampling_2_Layer_0");



    TCanvas* preWiggle = plotTProfile2D(hpre, "#Delta#eta(particle, cell)", "#Delta#phi(particle, cell)", "FCS V2 cell energy / Geant4 cell energy", "No accordion correction, EM Barrel 2", "pre_wiggle");

    TCanvas* postWiggle = plotTProfile2D(hpost, "#Delta#eta(particle, cell)", "#Delta#phi(particle, cell)", "FCS V2 cell energy / Geant4 cell energy", "Accordion correction, EM Barrel 2", "post_wiggle");


    preWiggle->Print("PreWiggleCorrection.pdf", "pdf");
    postWiggle->Print("PostWiggleCorrection.pdf", "pdf");

}


TCanvas* plotTGraph(TGraph* spline, TGraphErrors* gr, string xlabel, string ylabel, string label, string canvas)
{

    spline->SetLineColor(kBlack);

    spline->GetXaxis()->SetTitle(xlabel.c_str());
    spline->GetYaxis()->SetTitle(ylabel.c_str());



    gr->GetXaxis()->SetTitle(xlabel.c_str());
    gr->GetYaxis()->SetTitle(ylabel.c_str());

    gr->SetMarkerColor(kRed);
    gr->SetMarkerSize(1.5);

    TLegend* leg = new TLegend(0.7, 0.55, 0.9, 0.7);
    leg->SetBorderSize(0);
    leg->SetFillStyle(0);
    leg->SetFillColor(0);


    leg->AddEntry(gr, "Geant4", "pe");
    leg->AddEntry(spline, "Spline", "l");

    TCanvas * c1 = new TCanvas(canvas.c_str(), canvas.c_str(), 1200, 900);
    c1->cd();
    c1->SetLogx();
    spline->GetXaxis()->SetTitle(xlabel.c_str());
    spline->GetYaxis()->SetTitle(ylabel.c_str());

    gr->GetXaxis()->SetTitle(xlabel.c_str());
    gr->GetYaxis()->SetTitle(ylabel.c_str());

    spline->Draw("AL");
    gr->Draw("PE");
    leg->Draw();

    ATLASLabel(0.4, 0.2, "Simulation Internal");
    myText(0.2, 0.85, 1, label.c_str());

    gPad->Update();

    return c1;

}


void EnergyInterpolationPlot()
{

    TFile* fphoton = TFile::Open("/eos/atlas/user/a/ahasib/public/Simul-FastCalo/ParametrizationProductionVer02/mc16_13TeV.pid22.Einterpol.ver01.root");

    TFile* fpion = TFile::Open("/eos/atlas/user/a/ahasib/public/Simul-FastCalo/ParametrizationProductionVer02/mc16_13TeV.pid211.Einterpol.ver01.root");


    TCanvas* cphoton = (TCanvas*)fphoton->Get("EI_315_320_");
    TGraph* sp_ph = (TGraph*)cphoton->GetPrimitive("EI_315_320_");
    TGraphErrors* gr_ph = (TGraphErrors*)cphoton->GetPrimitive("Graph");

    TCanvas* cpion = (TCanvas*)fpion->Get("EI_15_20_");
    TGraph* sp_pi = (TGraph*)cpion->GetPrimitive("EI_15_20_");
    TGraphErrors* gr_pi = (TGraphErrors*)cpion->GetPrimitive("Graph");



    TCanvas* photon = plotTGraph(sp_ph, gr_ph, "Energy of the particle [MeV]", "Energy response", "#gamma, 3.15 < |#eta| < 3.20", "photon_canvas");

    TCanvas* pion = plotTGraph(sp_pi, gr_pi, "Energy of the particle [MeV]", "Energy response", "#pi^{#pm}, 0.15 < |#eta| < 0.20", "pion_canvas");


    photon->Print("EnergyInterpolationPhoton.pdf", "pdf");
    pion->Print("EnergyInterpolationPion.pdf", "pdf");





}

void ShapeValidation()
{

    gStyle->SetPalette(1);

    TFile *fshape = TFile::Open("/eos/atlas/atlascerngroupdisk/proj-simul/flatTTree/ShapeValidation.root");

    // TCanvas *can = (TCanvas*)fshape->Get("Shape2D_ShapeAvgShape_ratio_photon_E65536_eta020_025_cs02_allpca_");

    TCanvas *can = (TCanvas*)fshape->Get("photon_E65536_eta020_025_cs02_pca5_cellEvsdxdy_directratio_G4Input_AllSim_2D");


    TProfile2D* prof = (TProfile2D*)can->GetPrimitive("Clone_photon_E65536_eta020_025_cs02_pca5_cellEvsdxdy_directratio_G4Input_AllSim_2D");

    TCanvas* c = plotTProfile2D(prof, "#Delta#eta", "#Delta#phi", "Geant4 / FCS V2", "#gamma, 65 GeV, 0.20 < |#eta| < 0.25, EM Barrel 2", "shape_ratio_cs02_allpca");

    c->Print("shapeValidation.pdf", "pdf");
}

void G4ComparisonPlot(std::string fFCSV2Name = "", std::string fG4Name = " ")
{





    system("mkdir -p G4comparison_plot");


    bool is44GeVeta20 = 1;
    bool is65GeVeta20 = 1;
    bool is200GeVeta20 = 1;
    bool is65GeVeta55 = 1;
    bool is200GeVeta55 = 1;

    if (is44GeVeta20) {

        prefixEbin = "44GeV";
        prefixEta = "eta20_25";

        fFCSV2Name = "/eos/atlas/atlascerngroupdisk/proj-simul/flatTTree/rel21_0_70/FastCalo_simulation/photon_E43500_eta20_z0.root";

        fG4Name = "/eos/atlas/atlascerngroupdisk/proj-simul/flatTTree/rel21_0_70/Full_simulation/photon_E43500_eta20_z0.root";

        TFile* fFCSV2 = TFile::Open(fFCSV2Name.c_str());
        TFile* fG4 = TFile::Open(fG4Name.c_str());



        plotTH1(fFCSV2, fG4, "photon", "Reta", 20, 0.95, 0.99, "", "Reta = E(3#times7)/E(7#times7)", "Normalized to unity", "#gamma, 44 GeV, 0.20 < |#eta| < 0.25" );

        plotTH1(fFCSV2, fG4, "photon", "Rphi", 20, 0.94, 1, "", "Rphi = E(3#times3)/E(3#times7)", "Normalized to unity", "#gamma, 44 GeV, 0.20 < |#eta| < 0.25" );

        plotTH1(fFCSV2, fG4, "photon", "weta1", 30, 0.45, 0.7, "", "Weta = shower width in EM strip layer", "Normalized to unity", "#gamma, 44 GeV, 0.20 < |#eta| < 0.25" );



        fFCSV2->Close();
        fG4->Close();

    }

    if (is65GeVeta20) {

        prefixEbin = "65GeV";
        prefixEta = "eta20_25";

        fFCSV2Name = "/eos/atlas/atlascerngroupdisk/proj-simul/flatTTree/rel21_0_73r07/FastCalo_simulation/photon_E65536_eta20_z0/data-comparisonInput/photon_E65536_eta20_z0.root";

        fG4Name = "/eos/atlas/atlascerngroupdisk/proj-simul/flatTTree/rel21_0_73r07/Full_simulation/photon_E65536_eta20_z0/data-comparisonInput/photon_E65536_eta20_z0.root";

        TFile* fFCSV2 = TFile::Open(fFCSV2Name.c_str());
        TFile* fG4 = TFile::Open(fG4Name.c_str());



        plotTH1(fFCSV2, fG4, "photon", "Reta", 20, 0.95, 0.99, "", "Reta = E(3#times7)/E(7#times7)", "Normalized to unity", "#gamma, 65 GeV, 0.20 < |#eta| < 0.25" );

        plotTH1(fFCSV2, fG4, "photon", "Rphi", 20, 0.94, 1, "", "Rphi = E(3#times3)/E(3#times7)", "Normalized to unity", "#gamma, 65 GeV, 0.20 < |#eta| < 0.25" );

        plotTH1(fFCSV2, fG4, "photon", "weta1", 30, 0.45, 0.7, "", "Weta = shower width in EM strip layer", "Normalized to unity", "#gamma, 65 GeV, 0.20 < |#eta| < 0.25" );



        fFCSV2->Close();
        fG4->Close();

        fFCSV2Name = "/eos/atlas/atlascerngroupdisk/proj-simul/flatTTree/rel21_0_73r07/FastCalo_simulation/pion_E65536_eta55_z0/data-comparisonInput/pion_E65536_eta55_z0.root";

        fG4Name = "/eos/atlas/atlascerngroupdisk/proj-simul/flatTTree/rel21_0_73r07/Full_simulation/pion_E65536_eta55_z0/data-comparisonInput/pion_E65536_eta55_z0.root";


        fFCSV2 = TFile::Open(fFCSV2Name.c_str());
        fG4 = TFile::Open(fG4Name.c_str());

        // plotTH1(fFCSV2, fG4, "clusters", "N_clusters_inclusive", 7, 0.5, 7.5, "", "Number of clusters", "Normalized to unity", "#pi^{#pm}, 65 GeV, 0.20 < |#eta| < 0.25" , false, true);

        plotTH1(fFCSV2, fG4, "clusters", "N_clusters_inclusive", 7, 0.5, 7.5, "", "Number of clusters", "Normalized to unity", "#pi^{#pm}, 65 GeV, 0.20 < |#eta| < 0.25", true, true);


        plotTH1(fFCSV2, fG4, "clusters", "DeltaEta_1st", 30, 0, 0.03, "", "#Delta#eta (#pi, leading cluster)", "Normalized to unity", "#pi^{#pm}, 65 GeV, 0.20 < |#eta| < 0.25" );

        plotTH1(fFCSV2, fG4, "clusters", "LeadingCluster_pt", 35, 20, 90, "", " leading cluster p_{T} [GeV]", "Normalized to unity", "#pi^{#pm}, 65 GeV, 0.20 < |#eta| < 0.25" );

        plotTH1(fFCSV2, fG4, "clustersMoments", "SECOND_LAMBDA", 30, 0, 50000, "", " <#lambda^{2}> [cm^{2}] per cluster", "Normalized to unity", "#pi^{#pm}, 65 GeV, 0.20 < |#eta| < 0.25" );

        plotTH1(fFCSV2, fG4, "clustersMoments", "SECOND_R", 50, 0, 50000, "", " <r^{2}> [cm^{2}] per cluster", "Normalized to unity", "#pi^{#pm}, 65 GeV, 0.20 < |#eta| < 0.25" );

        fFCSV2->Close();
        fG4->Close();


    }



    if (is65GeVeta55) {

        prefixEbin = "65GeV";
        prefixEta = "eta55_60";

        fFCSV2Name = "/eos/atlas/atlascerngroupdisk/proj-simul/flatTTree/rel21_0_73r07/FastCalo_simulation/photon_E65536_eta55_z0/data-comparisonInput/photon_E65536_eta55_z0.root";

        fG4Name = "/eos/atlas/atlascerngroupdisk/proj-simul/flatTTree/rel21_0_73r07/Full_simulation/photon_E65536_eta55_z0/data-comparisonInput/photon_E65536_eta55_z0.root";

        TFile* fFCSV2 = TFile::Open(fFCSV2Name.c_str());
        TFile* fG4 = TFile::Open(fG4Name.c_str());



        plotTH1(fFCSV2, fG4, "photon", "Reta", 20, 0.95, 0.99, "", "Reta = E(3#times7)/E(7#times7)", "Normalized to unity", "#gamma, 65 GeV, 0.55 < |#eta| < 0.60" );

        plotTH1(fFCSV2, fG4, "photon", "Rphi", 20, 0.94, 1, "", "Rphi = E(3#times3)/E(3#times7)", "Normalized to unity", "#gamma, 65 GeV, 0.55 < |#eta| < 0.60" );

        plotTH1(fFCSV2, fG4, "photon", "weta1", 30, 0.45, 0.7, "", "Weta = shower width in EM strip layer", "Normalized to unity", "#gamma, 65 GeV, 0.55 < |#eta| < 0.60" );



        fFCSV2->Close();
        fG4->Close();

        fFCSV2Name = "/eos/atlas/atlascerngroupdisk/proj-simul/flatTTree/rel21_0_73r07/FastCalo_simulation/pion_E65536_eta20_z0/data-comparisonInput/pion_E65536_eta20_z0.root";

        fG4Name = "/eos/atlas/atlascerngroupdisk/proj-simul/flatTTree/rel21_0_73r07/Full_simulation/pion_E65536_eta20_z0/data-comparisonInput/pion_E65536_eta20_z0.root";


        fFCSV2 = TFile::Open(fFCSV2Name.c_str());
        fG4 = TFile::Open(fG4Name.c_str());

        // plotTH1(fFCSV2, fG4, "clusters", "N_clusters_inclusive", 7, 0.5, 7.5, "", "Number of clusters", "Normalized to unity", "#pi^{#pm}, 65 GeV, 0.55 < |#eta| < 0.60" );

        plotTH1(fFCSV2, fG4, "clusters", "N_clusters_inclusive", 7, 0.5, 7.5, "", "Number of clusters", "Normalized to unity", "#pi^{#pm}, 65 GeV, 0.55 < |#eta| < 0.60", true, true);


        plotTH1(fFCSV2, fG4, "clusters", "DeltaEta_1st", 15, 0, 0.03, "", "#Delta#eta (#pi, leading cluster)", "Normalized to unity", "#pi^{#pm}, 65 GeV, 0.55 < |#eta| < 0.60" );

        plotTH1(fFCSV2, fG4, "clusters", "LeadingCluster_pt", 35, 20, 90, "", " leading cluster p_{T} [GeV]", "Normalized to unity", "#pi^{#pm}, 65 GeV, 0.55 < |#eta| < 0.60" );

        plotTH1(fFCSV2, fG4, "clustersMoments", "SECOND_LAMBDA", 30, 0, 50000, "", " <#lambda^{2}> [cm^{2}] per cluster", "Normalized to unity", "#pi^{#pm}, 65 GeV, 0.55 < |#eta| < 0.60" );

        plotTH1(fFCSV2, fG4, "clustersMoments", "SECOND_R", 50, 0, 50000, "", " <r^{2}> [cm^{2}] per cluster", "Normalized to unity", "#pi^{#pm}, 65 GeV, 0.55 < |#eta| < 0.60" );

        fFCSV2->Close();
        fG4->Close();


    }



    if (is200GeVeta20) {

        prefixEbin = "200GeV";
        prefixEta = "eta20_25";

        fFCSV2Name = "/eos/atlas/atlascerngroupdisk/proj-simul/flatTTree/rel21_0_73r07/FastCalo_simulation/photon_E200000_eta20_z0/data-comparisonInput/photon_E200000_eta20_z0.root";

        fG4Name = "/eos/atlas/atlascerngroupdisk/proj-simul/flatTTree/rel21_0_73r07/Full_simulation/photon_E200000_eta20_z0/data-comparisonInput/photon_E200000_eta20_z0.root";

        TFile* fFCSV2 = TFile::Open(fFCSV2Name.c_str());
        TFile* fG4 = TFile::Open(fG4Name.c_str());



        plotTH1(fFCSV2, fG4, "photon", "Reta", 20, 0.95, 0.99, "", "Reta = E(3#times7)/E(7#times7)", "Normalized to unity", "#gamma, 200 GeV, 0.20 < |#eta| < 0.25" );

        plotTH1(fFCSV2, fG4, "photon", "Rphi", 20, 0.94, 1, "", "Rphi = E(3#times3)/E(3#times7)", "Normalized to unity", "#gamma, 200 GeV, 0.20 < |#eta| < 0.25" );

        plotTH1(fFCSV2, fG4, "photon", "weta1", 30, 0.45, 0.7, "", "Weta = shower width in EM strip layer", "Normalized to unity", "#gamma, 200 GeV, 0.20 < |#eta| < 0.25" );



        fFCSV2->Close();
        fG4->Close();

        fFCSV2Name = "/eos/atlas/atlascerngroupdisk/proj-simul/flatTTree/rel21_0_73r07/FastCalo_simulation/pion_E200000_eta20_z0/data-comparisonInput/pion_E200000_eta20_z0.root";

        fG4Name = "/eos/atlas/atlascerngroupdisk/proj-simul/flatTTree/rel21_0_73r07/Full_simulation/pion_E200000_eta20_z0/data-comparisonInput/pion_E200000_eta20_z0.root";


        fFCSV2 = TFile::Open(fFCSV2Name.c_str());
        fG4 = TFile::Open(fG4Name.c_str());

        // plotTH1(fFCSV2, fG4, "clusters", "N_clusters_inclusive", 7, 0.5, 7.5, "", "Number of clusters", "Normalized to unity", "#pi^{#pm}, 200 GeV, 0.20 < |#eta| < 0.25" );

        plotTH1(fFCSV2, fG4, "clusters", "N_clusters_inclusive", 7, 0.5, 7.5, "", "Number of clusters", "Normalized to unity", "#pi^{#pm}, 200 GeV, 0.20 < |#eta| < 0.25", true, true);


        plotTH1(fFCSV2, fG4, "clusters", "DeltaEta_1st", 30, 0, 0.03, "", "#Delta#eta (#pi, leading cluster)", "Normalized to unity", "#pi^{#pm}, 200 GeV, 0.20 < |#eta| < 0.25" );

        plotTH1(fFCSV2, fG4, "clusters", "LeadingCluster_pt", 40, 80, 240, "", " leading cluster p_{T} [GeV]", "Normalized to unity", "#pi^{#pm}, 200 GeV, 0.20 < |#eta| < 0.25" );

        plotTH1(fFCSV2, fG4, "clustersMoments", "SECOND_LAMBDA", 30, 0, 50000, "", " <#lambda^{2}> [cm^{2}] per cluster", "Normalized to unity", "#pi^{#pm}, 200 GeV, 0.20 < |#eta| < 0.25" );

        plotTH1(fFCSV2, fG4, "clustersMoments", "SECOND_R", 50, 0, 50000, "", " <r^{2}> [cm^{2}] per cluster", "Normalized to unity", "#pi^{#pm}, 200 GeV, 0.20 < |#eta| < 0.25" );

        fFCSV2->Close();
        fG4->Close();


    }


    if (is200GeVeta55) {

        prefixEbin = "200GeV";
        prefixEta = "eta55_60";

        fFCSV2Name = "/eos/atlas/atlascerngroupdisk/proj-simul/flatTTree/rel21_0_73r07/FastCalo_simulation/photon_E200000_eta55_z0/data-comparisonInput/photon_E200000_eta55_z0.root";

        fG4Name = "/eos/atlas/atlascerngroupdisk/proj-simul/flatTTree/rel21_0_73r07/Full_simulation/photon_E200000_eta55_z0/data-comparisonInput/photon_E200000_eta55_z0.root";

        TFile* fFCSV2 = TFile::Open(fFCSV2Name.c_str());
        TFile* fG4 = TFile::Open(fG4Name.c_str());



        plotTH1(fFCSV2, fG4, "photon", "Reta", 20, 0.95, 0.99, "", "Reta = E(3#times7)/E(7#times7)", "Normalized to unity", "#gamma, 200 GeV, 0.55 < |#eta| < 0.60" );

        plotTH1(fFCSV2, fG4, "photon", "Rphi", 20, 0.94, 1, "", "Rphi = E(3#times3)/E(3#times7)", "Normalized to unity", "#gamma, 200 GeV, 0.55 < |#eta| < 0.60" );

        plotTH1(fFCSV2, fG4, "photon", "weta1", 30, 0.45, 0.7, "", "Weta = shower width in EM strip layer", "Normalized to unity", "#gamma, 200 GeV, 0.55 < |#eta| < 0.60" );



        fFCSV2->Close();
        fG4->Close();

        fFCSV2Name = "/eos/atlas/atlascerngroupdisk/proj-simul/flatTTree/rel21_0_73r07/FastCalo_simulation/pion_E200000_eta55_z0/data-comparisonInput/pion_E200000_eta55_z0.root";

        fG4Name = "/eos/atlas/atlascerngroupdisk/proj-simul/flatTTree/rel21_0_73r07/Full_simulation/pion_E200000_eta55_z0/data-comparisonInput/pion_E200000_eta55_z0.root";


        fFCSV2 = TFile::Open(fFCSV2Name.c_str());
        fG4 = TFile::Open(fG4Name.c_str());

        // plotTH1(fFCSV2, fG4, "clusters", "N_clusters_inclusive", 7, 0.5, 7.5, "", "Number of clusters", "Normalized to unity", "#pi^{#pm}, 200 GeV, 0.55 < |#eta| < 0.60" );

        plotTH1(fFCSV2, fG4, "clusters", "N_clusters_inclusive", 7, 0.5, 7.5, "", "Number of clusters", "Normalized to unity", "#pi^{#pm}, 200 GeV, 0.55 < |#eta| < 0.60", true, true );


        plotTH1(fFCSV2, fG4, "clusters", "DeltaEta_1st", 30, 0, 0.03, "", "#Delta#eta (#pi, leading cluster)", "Normalized to unity", "#pi^{#pm}, 200 GeV, 0.55 < |#eta| < 0.60" );

        plotTH1(fFCSV2, fG4, "clusters", "LeadingCluster_pt", 20, 80, 240, "", " leading cluster p_{T} [GeV]", "Normalized to unity", "#pi^{#pm}, 200 GeV, 0.55 < |#eta| < 0.60" );

        plotTH1(fFCSV2, fG4, "clustersMoments", "SECOND_LAMBDA", 30, 0, 50000, "", " <#lambda^{2}> [cm^{2}] per cluster", "Normalized to unity", "#pi^{#pm}, 200 GeV, 0.55 < |#eta| < 0.60" );

        plotTH1(fFCSV2, fG4, "clustersMoments", "SECOND_R", 50, 0, 50000, "", " <r^{2}> [cm^{2}] per cluster", "Normalized to unity", "#pi^{#pm}, 200 GeV, 0.55 < |#eta| < 0.60" );

        fFCSV2->Close();
        fG4->Close();


    }




}

void runPlotApprovalICHEP()
{

    SetAtlasStyle();

    // shapePlot();
    // WiggleValidationPlot();
    EnergyInterpolationPlot();
    // G4ComparisonPlot();

    // ShapeValidation();

}
