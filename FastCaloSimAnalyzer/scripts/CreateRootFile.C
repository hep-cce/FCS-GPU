/*
  Copyright (C) 2002-2017 CERN for the benefit of the ATLAS collaboration
*/

#include "TROOT.h"
#include "TFile.h"
#include "TH2.h"
#include "TString.h"

#include <iostream>

void CreateRootFile() {


	std::vector<int> layer;
// std::vector<int> pca;

	layer.push_back(0);
	layer.push_back(1);
	layer.push_back(2);
	layer.push_back(3);
	layer.push_back(12);
	layer.push_back(13);
	layer.push_back(14);


	std::string outFile = "/Users/ahasib/Documents/Analysis/Projects/Athena/FCS_ShapeParametrization_dir/Simulation/ISF/ISF_FastCaloSim/run/output/shape_para/pionminus/InputDistributionNew_pionminus_50.000000GeV_eta_0.200000_0.250000_Norm.root";

	TFile *fout = new TFile(outFile.c_str(), "recreate");

	double total_energy = 0.;

	for (int i = 0; i < layer.size(); ++i)
	{
		for (int j = 1; j < 6; ++j)
		{
			std::string file = "/Users/ahasib/Documents/Analysis/Projects/Athena/FCS_ShapeParametrization_dir/Simulation/ISF/ISF_FastCaloSim/run/output/shape_para/pionminus/NNinputNew_pionminus_50.000000GeV_eta_0.200000_0.250000_layer" + std::to_string(layer.at(i)) + "_PCAbin" + std::to_string(j) + ".root";



			TFile * f = TFile::Open(file.c_str());



			//TH1F *h = (TH1F*)f->Get("hDrEnergy");



			// TH1F *h2 = (TH1F*)h->Clone(Form("hDrEnergy_layer%i_pca%i", layer.at(i), j));
			// h2->SetTitle(Form("hDrEnergy_layer%i_pca%i", layer.at(i), j));



			TH2F *h_LnEnergyDensity = (TH2F*)f->Get("hEnergyNorm");

			TH2F *h2 = (TH2F*)h_LnEnergyDensity->Clone(Form("hEnergy_layer%i_pca%i", layer.at(i), j));
			h2->SetTitle(Form("hEnergy_layer%i_pca%i", layer.at(i), j));

			total_energy += h2->Integral();

			cout << "* layer = " << layer.at(i) << " pca = " << j << " energy =" << h2->Integral() <<  endl;

			fout->cd();
			h2->Write();
			f->Close();
		}
	}

	fout->Close();


}
