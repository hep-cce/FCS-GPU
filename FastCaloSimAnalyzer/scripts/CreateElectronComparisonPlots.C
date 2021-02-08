/*
  Copyright (C) 2002-2017 CERN for the benefit of the ATLAS collaboration
*/

#include "TROOT.h"
#include "TFile.h"
#include "TH1.h"
#include "TString.h"
#include "TLegend.h"

#include <iostream>


void CreateElectronComparisonPlots() {


	gStyle->SetOptStat(0);
	gROOT->SetBatch(kTRUE);


	std::vector<int> layer;

	layer.push_back(0);
	layer.push_back(1);
	layer.push_back(2);
	layer.push_back(3);



	for (int i = 0; i < layer.size(); ++i)
	{
		for (int j = 1; j < 5; ++j)
		{
			std::string el1mmFile = "/Users/hasib/Documents/Analysis/FastCaloSim/ISF_FastCaloSim/run/output/shape_para/el_1mm/HitsAlphaDr_el_1mm_50.000000GeV_eta_0.200000_0.250000_layer" + std::to_string(layer.at(i)) + "_PCAbin" + std::to_string(j) + ".root";

			std::string eloptFile = "/Users/hasib/Documents/Analysis/FastCaloSim/ISF_FastCaloSim/run/output/shape_para/el_opt/HitsAlphaDr_el_opt_50.000000GeV_eta_0.200000_0.250000_layer" + std::to_string(layer.at(i)) + "_PCAbin" + std::to_string(j) + ".root";


			std::cout << " location of el 1mm file =" << el1mmFile << std::endl;
			std::cout << " location of el opt file =" << eloptFile << std::endl;


			std::string outDetaHits = "/Users/hasib/Documents/Analysis/FastCaloSim/ISF_FastCaloSim/run/output/shape_para/comparison_1mm_opt/hits_deta_layer" + std::to_string(layer.at(i)) + "_pca" + std::to_string(j) + ".png";

			std::string outDetaEnergy = "/Users/hasib/Documents/Analysis/FastCaloSim/ISF_FastCaloSim/run/output/shape_para/comparison_1mm_opt/energy_deta_layer" + std::to_string(layer.at(i)) + "_pca" + std::to_string(j) + ".png";

			std::string outDphiHits = "/Users/hasib/Documents/Analysis/FastCaloSim/ISF_FastCaloSim/run/output/shape_para/comparison_1mm_opt/hits_dphi_layer" + std::to_string(layer.at(i)) + "_pca" + std::to_string(j) + ".png";

			std::string outDphiEnergy = "/Users/hasib/Documents/Analysis/FastCaloSim/ISF_FastCaloSim/run/output/shape_para/comparison_1mm_opt/energy_dphi_layer" + std::to_string(layer.at(i)) + "_pca" + std::to_string(j) + ".png";


			TFile *f1mm = TFile::Open(el1mmFile.c_str());
			TFile *fopt = TFile::Open(eloptFile.c_str());


			std::cout << " f1mm = " << f1mm << std::endl;
			std::cout << " fopt = " << fopt << std::endl;


			TH1F *hdeta_1mm = (TH1F*)f1mm->Get("hdeta_corr_mm");
			TH1F *hdphi_1mm = (TH1F*)f1mm->Get("hdphi_corr_mm");
			TH1F *hdeta_opt = (TH1F*)fopt->Get("hdeta_corr_mm");
			TH1F *hdphi_opt = (TH1F*)fopt->Get("hdphi_corr_mm");

			TH1F *hEdeta_1mm = (TH1F*)f1mm->Get("hdetaE_corr_mm");
			TH1F *hEdphi_1mm = (TH1F*)f1mm->Get("hdphiE_corr_mm");
			TH1F *hEdeta_opt = (TH1F*)fopt->Get("hdetaE_corr_mm");
			TH1F *hEdphi_opt = (TH1F*)fopt->Get("hdphiE_corr_mm");


			hdeta_1mm->Scale(1 / hdeta_1mm->Integral());
			hdphi_1mm->Scale(1 / hdphi_1mm->Integral());

			hdeta_opt->Scale(1 / hdeta_opt->Integral());
			hdphi_opt->Scale(1 / hdphi_opt->Integral());


			hEdeta_1mm->Scale(1 / hEdeta_1mm->Integral());
			hEdphi_1mm->Scale(1 / hEdphi_1mm->Integral());

			hEdeta_opt->Scale(1 / hEdeta_opt->Integral());
			hEdphi_opt->Scale(1 / hEdphi_opt->Integral());




			hdeta_1mm->GetXaxis()->SetTitle("#delta#eta [mm]");
			hdeta_1mm->GetYaxis()->SetTitle("hits norm.");

			hdphi_1mm->GetXaxis()->SetTitle("#delta#phi [mm]");
			hdphi_1mm->GetYaxis()->SetTitle("hits norm.");

			hdeta_opt->GetXaxis()->SetTitle("#delta#eta [mm]");
			hdeta_opt->GetYaxis()->SetTitle("hits norm.");

			hdphi_opt->GetXaxis()->SetTitle("#delta#phi [mm]");
			hdphi_opt->GetYaxis()->SetTitle("hits norm.");

			hEdeta_1mm->GetXaxis()->SetTitle("#delta#eta [mm]");
			hEdeta_1mm->GetYaxis()->SetTitle("energy norm.");

			hEdphi_1mm->GetXaxis()->SetTitle("#delta#phi [mm]");
			hEdphi_1mm->GetYaxis()->SetTitle("energy norm.");

			hEdeta_opt->GetXaxis()->SetTitle("#delta#eta [mm]");
			hEdeta_opt->GetYaxis()->SetTitle("energy norm.");

			hEdphi_opt->GetXaxis()->SetTitle("#delta#phi [mm]");
			hEdphi_opt->GetYaxis()->SetTitle("energy norm.");


			hdeta_1mm->GetYaxis()->SetTitleOffset(1.4);
			hdphi_1mm->GetYaxis()->SetTitleOffset(1.4);
			hdeta_opt->GetYaxis()->SetTitleOffset(1.4);
			hdphi_opt->GetYaxis()->SetTitleOffset(1.4);

			hEdeta_1mm->GetYaxis()->SetTitleOffset(1.4);
			hEdphi_1mm->GetYaxis()->SetTitleOffset(1.4);
			hEdeta_opt->GetYaxis()->SetTitleOffset(1.4);
			hEdphi_opt->GetYaxis()->SetTitleOffset(1.4);


			if (layer.at(i) == 1) {
				hdeta_1mm->Rebin(1);
				hdeta_opt->Rebin(1);
			} else {
				hdeta_1mm->Rebin(5);
				hdphi_1mm->Rebin(5);
				hdeta_opt->Rebin(5);
				hdphi_opt->Rebin(5);

				hEdeta_1mm->Rebin(5);
				hEdphi_1mm->Rebin(5);
				hEdeta_opt->Rebin(5);
				hEdphi_opt->Rebin(5);
			}

			hdeta_1mm->GetXaxis()->SetRangeUser(-50, 50);
			hdphi_1mm->GetXaxis()->SetRangeUser(-50, 50);

			hdeta_opt->GetXaxis()->SetRangeUser(-50, 50);
			hdphi_opt->GetXaxis()->SetRangeUser(-50, 50);

			hEdeta_1mm->GetXaxis()->SetRangeUser(-50, 50);
			hEdphi_1mm->GetXaxis()->SetRangeUser(-50, 50);

			hEdeta_opt->GetXaxis()->SetRangeUser(-50, 50);
			hEdphi_opt->GetXaxis()->SetRangeUser(-50, 50);



			hdeta_1mm->SetLineColor(kRed);
			hdphi_1mm->SetLineColor(kRed);
			hdeta_opt->SetLineColor(kBlue);
			hdphi_opt->SetLineColor(kBlue);
			hEdeta_1mm->SetLineColor(kRed);
			hEdphi_1mm->SetLineColor(kRed);
			hEdeta_opt->SetLineColor(kBlue);
			hEdphi_opt->SetLineColor(kBlue);


			hdeta_1mm->SetTitle("");
			hdphi_1mm->SetTitle("");
			hdeta_opt->SetTitle("");
			hdphi_opt->SetTitle("");
			hEdeta_1mm->SetTitle("");
			hEdphi_1mm->SetTitle("");
			hEdeta_opt->SetTitle("");
			hEdphi_opt->SetTitle("");





			TCanvas * c1 = new TCanvas(Form("c1_layer%i_pca%i", layer.at(i), j), Form("c1_layer%i_pca%i", layer.at(i), j), 0, 0, 800, 700);
			TCanvas * c2 = new TCanvas(Form("c2_layer%i_pca%i", layer.at(i), j), Form("c2_layer%i_pca%i", layer.at(i), j), 0, 0, 800, 700);
			TCanvas * c3 = new TCanvas(Form("c3_layer%i_pca%i", layer.at(i), j), Form("c3_layer%i_pca%i", layer.at(i), j), 0, 0, 800, 700);
			TCanvas * c4 = new TCanvas(Form("c4_layer%i_pca%i", layer.at(i), j), Form("c4_layer%i_pca%i", layer.at(i), j), 0, 0, 800, 700);


			TLegend *l1 = new TLegend(0.75, 0.75, 0.85, 0.85);
			l1->SetLineColor(0);
			l1->AddEntry(hdeta_1mm, "1mm", "l");
			l1->AddEntry(hdeta_opt, "opt", "l" );

			TLegend *l2 = new TLegend(0.75, 0.75, 0.85, 0.85);
			l2->SetLineColor(0);
			l2->AddEntry(hdphi_1mm, "1mm", "l");
			l2->AddEntry(hdphi_opt, "opt", "l" );

			TLegend *l3 = new TLegend(0.75, 0.75, 0.85, 0.85);
			l3->SetLineColor(0);
			l3->AddEntry(hEdeta_1mm, "1mm", "l");
			l3->AddEntry(hEdeta_opt, "opt", "l" );

			TLegend *l4 = new TLegend(0.75, 0.75, 0.85, 0.85);
			l4->SetLineColor(0);
			l4->AddEntry(hEdphi_1mm, "1mm", "l");
			l4->AddEntry(hEdphi_opt, "opt", "l" );


			std::string label = "EMB" + std::to_string(layer.at(i)) + "   bin(PCA) = " + std::to_string(j);




			// TLatex *latex1 = new TLatex(15, 0.1, label.c_str());;

			TLatex *latex = new TLatex();
			latex->SetTextSize(.02);
			latex->SetTextFont(42);


			c1->cd();
			hdeta_1mm->Draw("hist");
			hdeta_opt->Draw("hist same");
			l1->Draw("same");
			float yval1 = hdeta_1mm->GetMaximum();
			latex->DrawLatex(15, yval1, label.c_str());
			c1->SaveAs(outDetaHits.c_str());



			c2->cd();
			hEdeta_opt->Draw("hist");
			hEdeta_1mm->Draw("hist same");
			l2->Draw("same");
			float yval2 = hEdeta_opt->GetMaximum();
			latex->DrawLatex(15, yval2, label.c_str());
			c2->SaveAs(outDetaEnergy.c_str());


			c3->cd();
			hdphi_1mm->Draw("hist");
			hdphi_opt->Draw("hist same");
			l3->Draw("same");
			float yval3 = hdphi_1mm->GetMaximum();
			latex->DrawLatex(15, yval3, label.c_str());
			c3->SaveAs(outDphiHits.c_str());


			c4->cd();
			hEdphi_opt->Draw("hist");
			hEdphi_1mm->Draw("hist same");
			l3->Draw("same");
			float yval4 = hEdphi_opt->GetMaximum();
			latex->DrawLatex(15, yval4, label.c_str());
			c4->SaveAs(outDphiEnergy.c_str());

			// c1->Close();
			// c2->Close();
			// c3->Close();
			// c4->Close();




		}
	}







}
