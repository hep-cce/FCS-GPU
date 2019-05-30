/*
  Copyright (C) 2002-2018 CERN for the benefit of the ATLAS collaboration
*/


#include "TFCSAnalyzerBase.h"
#include "TFile.h"
#include "TChain.h"
#include "TTree.h"
#include "TString.h"
#include "TH2.h"
#include "TH1.h"

#include <iostream>
#include <algorithm>
#include <string>
#include <stdlib.h>
#include <tuple>



string prefixlayer;



TH1F* GetEfficiencyHistogram(TH1F* htot, TH1F* hmatch) {

    std::string title = hmatch->GetName();
    title = title + "_efficiency";
    TH1F* heff = (TH1F*)hmatch->Clone();
    heff->Divide(htot);
    heff->GetXaxis()->SetTitle("#phi_{hit} - #phi_{cell}");
    heff->SetName(title.c_str());
    heff->SetTitle(title.c_str());

    return heff;

}

TCanvas* plotWiggle(TH1F* htot, TH1F* hmatch, TH1F* heff, TH1F* hderiv, TH1F* hderivpos, std::string outDir) {

    std::string title = hmatch->GetName();
    title = title + "_wiggle_correction";

    system(("mkdir -p " + outDir).c_str());
    std::string outfile = outDir + title;


    TCanvas * c = new TCanvas(title.c_str(), title.c_str(), 0, 0, 1200, 1000);
    c->Divide(3, 2);
    c->cd(1);
    htot->Draw();
    c->cd(2);
    hmatch->Draw();
    c->cd(3);
    heff->Draw();
    c->cd(4);
    hderiv->Draw();
    c->cd(5);
    hderivpos->Draw();

    c->SaveAs((outfile + ".png").c_str());

    return c;


}

TH1F* GetDerivative(TH1F* h) {

    std::string title = h->GetName();
    title = title + "_derivative";

    TH1F* h_deriv = (TH1F*)h->Clone();
    h_deriv->Reset();
    h_deriv->SetNameTitle(title.c_str(), title.c_str());

    int nbinsX = h->GetNbinsX();

    double deriv_i = -99;

    for (int i = 1; i < nbinsX / 2 + 1; i++) {

        if (i == 1 and h->GetBinContent(nbinsX) == 0) deriv_i = 0;
        else if (i == 1 and h->GetBinContent(nbinsX) != 0) deriv_i = (h->GetBinContent(i + 1) - h->GetBinContent(nbinsX)) / (2 * h->GetBinWidth(i));
        else if (h->GetBinContent(i + 1) == 0 or h->GetBinContent(i - 1) == 0) deriv_i = 0;
        else deriv_i = (h->GetBinContent(i + 1) - h->GetBinContent(i - 1)) / (2 * h->GetBinWidth(i));

        h_deriv->SetBinContent(nbinsX / 2 + 1 - i, deriv_i);
    }


    for (int i = nbinsX / 2 + 1; i < nbinsX + 1; i++) {
        if (h->GetBinContent(i + 1) == 0 or h->GetBinContent(i - 1) == 0) deriv_i = 0;
        else if (i == nbinsX and h->GetBinContent(nbinsX) == 1) deriv_i = 0;
        else if (i == nbinsX and h->GetBinContent(1) != 0) deriv_i = (h->GetBinContent(1) - h->GetBinContent(i - 1)) / (2 * h->GetBinWidth(i));
        else deriv_i = (h->GetBinContent(i + 1) - h->GetBinContent(i - 1)) / (2 * h->GetBinWidth(i));

        h_deriv->SetBinContent(nbinsX + 1 - (i - nbinsX / 2), -deriv_i);

    }

    return h_deriv;
}

void GetPositiveSplit(TH1F*& h) {
// normalize only one of half of the derivative histogram
    // std::cout << " integral of original derivative histogram = " << h->Integral() << std::endl;
    while (h->GetBinContent(h->GetMinimumBin()) < 0.) {

        int bmin = h->GetMinimumBin();
        int nbins = h->GetNbinsX();

        // calculate the distance for the first positive bin in both direction from the min

        float dist_right = -99;
        int loc_right = -1;
        float dist_left = -99;
        int loc_left = -1;

        for (int i = bmin + 1; i < nbins + 1; i++) {
            if (h->GetBinContent(i) > 0) {
                dist_right = fabs(i - bmin);
                loc_right = i;
                break;
            }
        }

        for (int i = bmin - 1; i > 0; i--) {
            if (h->GetBinContent(i) > 0) {
                dist_left = fabs(bmin - i);
                loc_left = i;
                break;
            }
        }


        // set bin content based on the distance of the left/right positive bin

        if (dist_right < dist_left) {
            h->SetBinContent(loc_right, h->GetBinContent(loc_right) + h->GetBinContent(bmin));
            h->SetBinContent(bmin, 0.);

        } else if (dist_left < dist_right) {
            h->SetBinContent(loc_left, h->GetBinContent(loc_left) + h->GetBinContent(bmin));
            h->SetBinContent(bmin, 0.);
        } else if (dist_left == dist_right) {

            h->SetBinContent(loc_right, h->GetBinContent(loc_right) + h->GetBinContent(bmin) / 2);
            h->SetBinContent(loc_left, h->GetBinContent(loc_left) + h->GetBinContent(bmin) / 2);
            h->SetBinContent(bmin, 0.);
        }


    }// end of while

    // std::cout << " integral of final derivative histogram = " << h->Integral() << std::endl;

}



TH1F* GetPositiveDerivative(TH1F* h) {
// split the derivative in two halves and normalize each half
// keep any possible asymmetry between each half

    std::string title = h->GetName();
    title = title + "_positive";

    int nbins = h->GetNbinsX();
    float xmin = h->GetBinLowEdge(1);
    float xmid = h->GetBinLowEdge(nbins / 2 + 1);
    float xmax = h->GetBinLowEdge(nbins + 1);
    TH1F * h1 = new TH1F((title + "_1").c_str(), (title + "_1").c_str(), nbins / 2, xmin, xmid);
    TH1F * h2 = new TH1F((title + "_2").c_str(), (title + "_2").c_str(), nbins / 2, xmid, xmax);


    for (int i = 1; i < nbins / 2; i++) {

        h1->SetBinContent(i, h->GetBinContent(i));
        h2->SetBinContent(i, h->GetBinContent(i + nbins / 2));

    }

    GetPositiveSplit(h1);
    GetPositiveSplit(h2);

    TH1F* h_deriv_pos = (TH1F*)h->Clone();
    h_deriv_pos->Reset();
    h_deriv_pos->SetNameTitle(title.c_str(), title.c_str());

    for (int i = 1; i < h1->GetNbinsX(); i++) {
        h_deriv_pos->SetBinContent(i, h1->GetBinContent(i));
    }


    for (int i = 1; i < h2->GetNbinsX(); i++) {
        h_deriv_pos->SetBinContent(i + nbins / 2, h2->GetBinContent(i));
    }

    return h_deriv_pos;

}




void runTFCSWiggleDerivativeHistograms(int dsid = 431004, int dsid_zv0 = -999,  std::string sampleData = "../python/inputSampleList.txt", std::string topDir = "./wiggle_output/", std::string version = "ver01", std::string topPlotDir = "wiggle_plot/")

{

    gROOT->SetBatch(kTRUE);




    /////////////////////////////
    // read smaple information
    // based on DSID
    //////////////////////////

    if (dsid_zv0 < 0)dsid_zv0 = dsid;

    TFCSAnalyzerBase::SampleInfo sample;
    sample = TFCSAnalyzerBase::GetInfo(sampleData.c_str(), dsid);

    std::string input = sample.inputSample;
    std::string baselabel = sample.label;
    int pdgid = sample.pdgid;
    int energy = sample.energy;
    float etamin = sample.etamin;
    float etamax = sample.etamax;
    int zv = sample.zv;


    TFCSAnalyzerBase::SampleInfo sample_zv = TFCSAnalyzerBase::GetInfo(sampleData.c_str(), dsid);




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
    TString pcaSample(Form("%s%s.firstPCA.%s.root", topDir.c_str(), baselabel.c_str(), version.c_str()));
    TString plotDir(Form("%s/%s.plots.%s/", topPlotDir.c_str(), baselabel.c_str(), version.c_str()));
    TString wiggleSample(Form("%s%s.wiggleDerivative.%s.root", topDir.c_str(), baselabel.c_str(), version.c_str()));

    system(("mkdir -p " + topDir).c_str());


    TString pcaAppSample = pcaSample;
    pcaAppSample.ReplaceAll("firstPCA", "firstPCA_App");


    /////////////////////////////////////////
    // read input sample and create first pca
    ///////////////////////////////////////////
    TString inputSample_zv = sample.inputSample.c_str();


    TChain * inputChain = new TChain("FCS_ParametrizationInput");
    inputChain->Add(inputSample_zv);

    int nentries = inputChain->GetEntries();

    std::cout << " * Prepare to run on: " << inputSample << " with entries = " << nentries << std::endl;


    TFCSMakeFirstPCA *myfirstPCA = new TFCSMakeFirstPCA(inputChain, pcaSample.Data());
    myfirstPCA->set_cumulativehistobins(5000);
    myfirstPCA->set_edepositcut(0.001);
    myfirstPCA->apply_etacut(0);
    myfirstPCA->run();
    delete myfirstPCA;
    cout << "TFCSMakeFirstPCA done" << endl;



    int npca1 = 5;
    int npca2 = 1;




    TFCSApplyFirstPCA *myfirstPCA_App = new TFCSApplyFirstPCA(pcaSample.Data());
    myfirstPCA_App->set_pcabinning(npca1, npca2);
    myfirstPCA_App->init();
    myfirstPCA_App->run_over_chain(inputChain, pcaAppSample.Data());
    delete myfirstPCA_App;
    cout << "TFCSApplyFirstPCA done" << endl;





// -------------------------------------------------------


    TChain *pcaChain = new TChain("tree_1stPCA");
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
    for (int ibiny = 1; ibiny <= relevantLayers->GetNbinsY(); ibiny++ )
    {
        if ( relevantLayers->GetBinContent(1, ibiny) == 1) v_layer.push_back(ibiny - 1);
    }

    std::cout << " relevantLayers = ";
    for (auto i : v_layer) std::cout << i << " ";
    std::cout << "\n";


    //////////////////////////////////////////////////////////
    ///// Create validation steering
    //////////////////////////////////////////////////////////



    TFile* fwiggle  = new TFile(wiggleSample, "recreate");


    for (int ilayer = 0; ilayer < v_layer.size(); ilayer++) {

        int analyze_layer = v_layer.at(ilayer);

        if (analyze_layer > 7) {
            std::cout << "Skipping layers that doesn't belong to the EM calorimeter" << std::endl;
            continue;
        }

        // phi granularity is different for different layers
        // in additon for layers 1 and 6 the phi granularity changes at some eta: needs separate wiggle efficiency functions below and above this eta boundary

        const float pi = TMath::Pi();
        float phi_gran[8] = {pi / 32, pi / 32, pi / 128, pi / 128, pi / 32, pi / 32, pi / 128, pi / 128};

        float eta_boundary = -1;
        float phi_gran_above_etaboundary = -1;
        if (analyze_layer == 1) {
            eta_boundary = 1.4;
            phi_gran_above_etaboundary = pi / 128;
        }
        else if (analyze_layer == 6) {
            eta_boundary = 2.5;
            phi_gran_above_etaboundary = pi / 32;
        }


        TH1F* h_total_dphi = nullptr;
        TH1F* h_matched_dphi = nullptr;
        TH1F* h_total_dphi_etaboundary = nullptr;
        TH1F* h_matched_dphi_etaboundary = nullptr;




        TFCSShapeValidation analyze(inputChain, analyze_layer);
        analyze.set_IsNewSample(true);
        analyze.set_Nentries(nentries);
        //analyze.set_Nentries(500);
        analyze.set_Debug(0);
        int analyze_pcabin = -1; // do not separate in pca bins

        prefixlayer = Form("cs%d", analyze_layer);
        // prefixEbin = Form("pca%d_", analyze_pcabin);


        std::cout << "=============================" << std::endl;

        //////////////////////////////////////////////////////////
        ///// Chain to read the hits from the input file
        //////////////////////////////////////////////////////////


        TFCSParametrizationChain* RunInputHits = new TFCSParametrizationChain("input_EnergyAndHits", "original energy and hits from input file");




        TFCSValidationEnergyAndHits* input_EnergyAndHits = new TFCSValidationEnergyAndHits("input_EnergyAndHits", "original energy and hits from input file", &analyze);


        input_EnergyAndHits->set_pdgid(pdgid);
        input_EnergyAndHits->set_calosample(analyze_layer);
        input_EnergyAndHits->set_Ekin_bin(analyze_pcabin);
        input_EnergyAndHits->Print();

        RunInputHits->push_back(input_EnergyAndHits);
        RunInputHits->Print();

        TFCSValidationHitSpy* hitspy_orig = new TFCSValidationHitSpy("wiggle_efficiency", "accordion correcton");

        hitspy_orig->set_calosample(analyze_layer);
        hitspy_orig->set_eta_boundary(eta_boundary);
        hitspy_orig->set_previous(&input_EnergyAndHits->get_hitspy()); // this is required to retrived input cells in hitspy



        // get the total and matched histograms
        int nbins = 200;
        float phi = 0.5 * phi_gran[analyze_layer];


        h_total_dphi = analyze.InitTH1(prefixlayer, "total_1D", nbins, -phi, phi);
        hitspy_orig->hist_total_dphi() = h_total_dphi;
        h_total_dphi->SetName(prefixlayer.c_str());
        h_matched_dphi = analyze.InitTH1(prefixlayer, "matched_1D", nbins, -phi, phi);
        hitspy_orig->hist_matched_dphi() = h_matched_dphi;
        h_matched_dphi->SetName(prefixlayer.c_str());

        // for these two layers we have another extra histogram
        if (analyze_layer == 1 or analyze_layer == 6) {

            phi = 0.5 * phi_gran_above_etaboundary;

            h_total_dphi_etaboundary = analyze.InitTH1((prefixlayer + "above_etaboundary").c_str(), "total_1D", nbins, -phi, phi);
            hitspy_orig->hist_total_dphi_etaboundary() = h_total_dphi_etaboundary;
            h_total_dphi_etaboundary->SetName((prefixlayer + "_above_etaboundary").c_str());
            h_matched_dphi_etaboundary = analyze.InitTH1((prefixlayer + "above_etaboundary").c_str(), "matched_1D", nbins, -phi, phi);
            hitspy_orig->hist_matched_dphi_etaboundary() = h_matched_dphi_etaboundary;
            h_matched_dphi_etaboundary->SetName((prefixlayer + "_above_etaboundary").c_str());

        }

        input_EnergyAndHits->push_back(hitspy_orig);
        analyze.validations().emplace_back(RunInputHits);





        std::cout << "=============================" << std::endl;
        //////////////////////////////////////////////////////////
        analyze.LoopEvents(-1);


        TH1F* h_eff_below_etaboundary = nullptr;
        TH1F* h_eff_above_etaboundary =  nullptr;
        TH1F* h_derivative_below_etaboundary = nullptr;
        TH1F* h_derivative_above_etaboundary = nullptr;
        TH1F* h_derivative_pos_below_etaboundary = nullptr;
        TH1F* h_derivative_pos_above_etaboundary = nullptr;

        h_eff_below_etaboundary = GetEfficiencyHistogram(h_total_dphi, h_matched_dphi);
        h_derivative_below_etaboundary = GetDerivative(h_eff_below_etaboundary);
        h_derivative_pos_below_etaboundary = GetPositiveDerivative(h_derivative_below_etaboundary);

        TCanvas* c_below_etaboundary  = nullptr;
        TCanvas* c_above_etaboundary = nullptr;

        c_below_etaboundary = plotWiggle(h_total_dphi, h_matched_dphi, h_eff_below_etaboundary, h_derivative_below_etaboundary, h_derivative_pos_below_etaboundary, plotDir.Data());

        // for layers where phi granularity changes
        if (h_total_dphi_etaboundary and h_matched_dphi_etaboundary) {
            h_eff_above_etaboundary = GetEfficiencyHistogram(h_total_dphi_etaboundary, h_matched_dphi_etaboundary);
            h_derivative_above_etaboundary = GetDerivative(h_eff_above_etaboundary);
            h_derivative_pos_above_etaboundary = GetPositiveDerivative(h_derivative_above_etaboundary);
            c_above_etaboundary = plotWiggle(h_total_dphi_etaboundary, h_matched_dphi_etaboundary, h_eff_above_etaboundary, h_derivative_above_etaboundary, h_derivative_pos_above_etaboundary, plotDir.Data());
        }


        fwiggle->cd();

        c_below_etaboundary->Write();
        if (c_above_etaboundary) c_above_etaboundary->Write();

        h_derivative_pos_below_etaboundary->Write();
        if (h_derivative_pos_above_etaboundary) h_derivative_pos_above_etaboundary->Write();

        if (h_total_dphi) delete h_total_dphi;
        if (h_matched_dphi) delete h_matched_dphi;
        if (h_total_dphi_etaboundary) delete h_total_dphi_etaboundary;
        if (h_matched_dphi_etaboundary) delete h_matched_dphi_etaboundary;
        if (h_eff_below_etaboundary) delete h_eff_below_etaboundary;
        if (h_eff_above_etaboundary) delete h_eff_above_etaboundary;
        if (h_derivative_below_etaboundary) delete h_derivative_below_etaboundary;
        if (h_derivative_above_etaboundary) delete h_derivative_above_etaboundary;
        if (h_derivative_pos_below_etaboundary) delete h_derivative_pos_below_etaboundary;
        if (h_derivative_pos_above_etaboundary) delete h_derivative_pos_above_etaboundary;
        if (c_below_etaboundary) delete c_below_etaboundary;
        if (c_above_etaboundary) delete c_above_etaboundary;

    }

    fwiggle->Close();




}
