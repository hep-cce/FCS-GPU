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
#include "TH1.h"

#include <iostream>
#include <algorithm>
#include <string>
#include <stdlib.h>
#include <tuple>



string prefixlayer;
string prefixEbin;
string prefixall;


TH1F* zoomHisto(TH1* h_in)
{

    double min = -999.;
    double max = 999.;

    double rmin = -999.;
    double rmax = 999.;

    TFCSAnalyzerBase::autozoom(h_in, min, max, rmin, rmax );

    // cout << "min, max, rmin, rmax = " << min << ", " << max << ", " << rmin << ", " << rmax << endl;

    int Nbins;
    int bins = 0;
    for (int b = h_in->FindBin(min); b <= h_in->FindBin(max); b++)
        bins++;
    Nbins = bins;

    int start = h_in->FindBin(min) - 1;

    TH1F* h_out = new TH1F(h_in->GetName() + TString("_zoom"), h_in->GetTitle(), Nbins, rmin, rmax);
    h_out->SetXTitle(h_in->GetXaxis()->GetTitle());
    h_out->SetYTitle(h_in->GetYaxis()->GetTitle());
    for (int b = 1; b <= h_out->GetNbinsX(); b++)
    {
        h_out->SetBinContent(b, h_in->GetBinContent(start + b));
        h_out->SetBinError(b, h_in->GetBinError(start + b));
    }

    return h_out;

}




void runTFCSMaxrz(int dsid = 431004, int dsid_zv0 = -999,  std::string sampleData = "../python/inputSampleList.txt", std::string topDir = "./output/", std::string version = "ver01", std::string topPlotDir = "output_plot/")

{

    system(("mkdir -p " + topDir).c_str());

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
    TString extrapolSample(Form("%s%s.extrapol.%s.root", topDir.c_str(), baselabel.c_str(), version.c_str()));
    TString plotDir(Form("%s/%s.plots.%s/", topPlotDir.c_str(), baselabel.c_str(), version.c_str()));

    TString pcaAppSample = pcaSample;
    pcaAppSample.ReplaceAll("firstPCA", "firstPCA_App");


    /////////////////////////////////////////
    // read input sample and create first pca
    ///////////////////////////////////////////
    TString inputSample_zv = sample.inputSample.c_str();


    TChain * inputChain = new TChain("FCS_ParametrizationInput");
    inputChain->Add(inputSample_zv);

    int nentries = inputChain->GetEntries();
    // nentries = 15;

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

    // pcaSample = Form("../../EnergyParametrization/scripts/output/ds%i.FirstPCA.ver01.root",dsid_zv0);


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

    // v_layer.clear();
    // v_layer.push_back(2);
    // npca = 1;


    TFile* f = new TFile(extrapolSample, "recreate");

    for (int ilayer = 0; ilayer < v_layer.size(); ilayer++) {

        vector<TH2*> h_orig_hitEnergy_alpha_r(npca);
        vector<TH1F*> h_deltaEtaAveragedPerEvent(npca);

        vector<TH1F*> h_hitenergy_r(npca);
        vector<TH1F*> h_hitenergy_z(npca);

        vector<TH1F*> h_orig_max_hitenergy_r(npca);
        vector<TH1F*> h_orig_max_hitenergy_z(npca);


        vector<TFCSParametrizationChain*> RunInputHits(npca);
        vector<TFCSValidationEnergyAndHits*> input_EnergyAndHits(npca);
        vector<TFCSValidationHitSpy*> hitspy_orig(npca);

        int analyze_layer = v_layer.at(ilayer);
        TFCSShapeValidation analyze(inputChain, analyze_layer);
        analyze.set_IsNewSample(true);
        analyze.set_Nentries(nentries);
        analyze.set_Debug(0);

        for (int ipca = 1; ipca <= npca; ipca++) {

            int i = ipca - 1;
            int analyze_pcabin = ipca;

            prefixlayer = Form("cs%d_", analyze_layer);
            prefixall = Form("cs%d_pca%d_", analyze_layer, analyze_pcabin);
            prefixEbin = Form("pca%d_", analyze_pcabin);





            std::cout << "=============================" << std::endl;

            //////////////////////////////////////////////////////////
            ///// Chain to read 2D alpha_radius in mm from the input file
            //////////////////////////////////////////////////////////


            RunInputHits[i] = new TFCSParametrizationChain("input_EnergyAndHits", "original energy and hits from input file");


            input_EnergyAndHits[i] = new TFCSValidationEnergyAndHits("input_EnergyAndHits", "original energy and hits from input file", &analyze);


            input_EnergyAndHits[i]->set_pdgid(pdgid);
            input_EnergyAndHits[i]->set_calosample(analyze_layer);
            input_EnergyAndHits[i]->set_Ekin_bin(analyze_pcabin);

            RunInputHits[i]->push_back(input_EnergyAndHits[i]);
            RunInputHits[i]->Print();


            hitspy_orig[i] = new TFCSValidationHitSpy("hitspy_2D_E_alpha_radius", "shape parametrization");


            hitspy_orig[i]->set_calosample(analyze_layer);
            RunInputHits[i]->push_back(hitspy_orig[i]); // to call the simulate() method in HitSpy

            int binwidth = 5;
            if (analyze_layer == 1 or analyze_layer == 5)
                binwidth = 1;
            float rmin = 0;
            float rmax = 30000;
            int nbinsr = (int)((rmax - rmin) / binwidth);

            float zmin = -10000;
            float zmax = -zmin;
            int nbinsz = (int)((zmax - zmin) / binwidth);



            h_hitenergy_r[i] = analyze.InitTH1(prefixall + "hist_hitenergy_R", "1D", nbinsr, rmin, rmax);
            hitspy_orig[i]->hist_hitenergy_r() = h_hitenergy_r[i];
            h_orig_max_hitenergy_r[i] = analyze.InitTH1(prefixall + "hist_max_hitenergy_R", "1D", nbinsr, rmin, rmax);
            hitspy_orig[i]->hist_hitenergy_max_r() = h_orig_max_hitenergy_r[i];

            h_hitenergy_z[i] = analyze.InitTH1(prefixall + "hist_hitenergy_z", "1D", nbinsz, zmin, zmax);
            hitspy_orig[i]->hist_hitenergy_z() = h_hitenergy_z[i];
            h_orig_max_hitenergy_z[i] = analyze.InitTH1(prefixall + "hist_max_hitenergy_z", "1D", nbinsz, zmin, zmax);
            hitspy_orig[i]->hist_hitenergy_max_z() = h_orig_max_hitenergy_z[i];



            input_EnergyAndHits[i]->push_back(hitspy_orig[i]);
            analyze.validations().emplace_back(RunInputHits[i]);



        }

        std::cout << "=============================" << std::endl;
        //////////////////////////////////////////////////////////
        analyze.LoopEvents(-1);

        for (int ipca = 1; ipca <= npca; ipca++) {
            int i = ipca - 1;



            f->cd();

            TH1F* h_zoom_max_r = zoomHisto(h_orig_max_hitenergy_r[i]);
            TH1F* h_zoom_max_z = zoomHisto(h_orig_max_hitenergy_z[i]);

            h_zoom_max_r->Write();
            h_zoom_max_z->Write();



            delete h_zoom_max_r;
            delete h_zoom_max_z;
            delete h_orig_max_hitenergy_r[i];
            delete h_orig_max_hitenergy_z[i];


            delete RunInputHits[i];
            delete input_EnergyAndHits[i];
            delete hitspy_orig[i];

        }

    }

    f->Close();

}
