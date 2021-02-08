/*
  Copyright (C) 2002-2017 CERN for the benefit of the ATLAS collaboration
*/

#include "TROOT.h"
#include "TFile.h"
#include "TH2.h"
#include "TString.h"
#include "TLatex.h"

#include <iostream>

void CreateHitEtaPhiPlots(std::string particle, bool isNewSample) {

	gStyle->SetOptStat(0);
	gROOT->SetBatch(kTRUE);


	std::vector<int> layer;
	std::vector<int> pca;

	layer.push_back(0);
	layer.push_back(1);
	layer.push_back(2);
	layer.push_back(3);
	layer.push_back(12);
	//layer.push_back(13);
	//layer.push_back(14);

	pca.push_back(99);
	pca.push_back(1);
	pca.push_back(2);
	pca.push_back(3);
	pca.push_back(4);
	pca.push_back(5);


	std::string prefix = "";
	if (isNewSample) prefix = "HitsNewAlphaDr";
	else prefix = "HitsAlphaDr";

	std::string outFile = "";

	if (isNewSample) {
		outFile = "/Users/ahasib/Documents/Analysis/Projects/Athena/FCS_ShapeParametrization_dir/Simulation/ISF/ISF_FastCaloSim/run/output/shape_para/" + particle + "/PlotsNewHitEtaPhi_" + particle + "_50.000000GeV_eta_0.200000_0.250000/";
	} else {
		outFile = "/Users/ahasib/Documents/Analysis/Projects/Athena/FCS_ShapeParametrization_dir/Simulation/ISF/ISF_FastCaloSim/run/output/shape_para/" + particle + "/PlotsHitEtaPhi_" + particle + "_50.000000GeV_eta_0.200000_0.250000/";

	}

	// TFile *fout = new TFile(outFile.c_str(), "recreate");

	system(("mkdir -p " + outFile + "eta/").c_str());
	system(("mkdir -p " + outFile + "phi/").c_str());


	for (int i = 0; i < layer.size(); ++i)
	{
		for (int j = 0; j < pca.size(); j++)
		{
			std::string file = "/Users/ahasib/Documents/Analysis/Projects/Athena/FCS_ShapeParametrization_dir/Simulation/ISF/ISF_FastCaloSim/run/output/shape_para/" + particle + "/" + prefix + "_" + particle + "_50.000000GeV_eta_0.200000_0.250000_layer" + std::to_string(layer.at(i)) + "_PCAbin" + std::to_string(pca.at(j)) + ".root";



			TFile * f = TFile::Open(file.c_str());



			//TH1F *h = (TH1F*)f->Get("hDrEnergy");



			// TH1F *h2 = (TH1F*)h->Clone(Form("hDrEnergy_layer%i_pca%i", layer.at(i), j));
			// h2->SetTitle(Form("hDrEnergy_layer%i_pca%i", layer.at(i), j));


			TLatex *l = new TLatex(0.2 , 0.8, "ATLAS");
			// l->SetTextSize(.035);
			// l->SetTextFont(72);

			TLatex *l2 = new TLatex(0.4, 0.8, "Simulation Internal");
			// l2->SetTextSize(.035);
			// l2->SetTextFont(42);

			string pca_label = "";
			if (pca.at(j) == 99) pca_label = "incl.";
			else pca_label = std::to_string(pca.at(j));

			std::string labeltitle = "Layer :" + std::to_string(layer.at(i)) + " PCA : " + pca_label;

			TLatex * lInputTitle = new TLatex(-0.2, 100, labeltitle.c_str());
			lInputTitle->SetTextSize(.03);
			lInputTitle->SetTextFont(42);

			TH1F *heta_hit = (TH1F*)f->Get("heta_hit");
			heta_hit->Scale(1 / heta_hit->Integral());
			heta_hit->SetTitle(labeltitle.c_str());
			heta_hit->GetXaxis()->SetTitle("#eta-hit");
			heta_hit->GetYaxis()->SetTitle("a.u.");


			TH1F *hphi_hit = (TH1F*)f->Get("hphi_hit");
			hphi_hit->Rebin(8);
			hphi_hit->SetTitle(labeltitle.c_str());
			hphi_hit->Scale(1 / hphi_hit->Integral());
			hphi_hit->GetXaxis()->SetTitle("#phi-hit");
			hphi_hit->GetYaxis()->SetTitle("a.u");


			TCanvas * c1 = new TCanvas(("heta_hit_layer" + std::to_string(layer.at(i)) + "_pca" + std::to_string(pca.at(j))).c_str(), ("heta_hit_layer" + std::to_string(layer.at(i)) + "_pca" + std::to_string(pca.at(j))).c_str(), 0, 0, 900, 900);

			TCanvas * c2 = new TCanvas(("hphi_hit_layer" + std::to_string(layer.at(i)) + "_pca" + std::to_string(pca.at(j))).c_str(), ("hphi_hit_layer" + std::to_string(layer.at(i)) + "_pca" + std::to_string(pca.at(j))).c_str(), 0, 0, 900, 900);


			c1->cd();
			heta_hit->Draw("hist same E1");
			// l->Draw();
			// l2->Draw();
			// lInputTitle->Draw();


			c2->cd();
			hphi_hit->Draw("hist same E1");
			// l->Draw();
			// l2->Draw();
			// lInputTitle->Draw();







			c1->SaveAs((outFile + "eta/" + "heta_hit_layer" + std::to_string(layer.at(i)) + "_pca" + std::to_string(pca.at(j)) + ".png").c_str());

			c2->SaveAs((outFile + "phi/" + "hphi_hit_layer" + std::to_string(layer.at(i)) + "_pca" + std::to_string(pca.at(j)) + ".png").c_str());




			// fout->cd();
			// h2->Write();
			// f->Close();
		}
	}

	// fout->Close();


}
