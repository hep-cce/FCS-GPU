/*
  Copyright (C) 2002-2017 CERN for the benefit of the ATLAS collaboration
*/

// this macro plots RMS

#include "TROOT.h"
#include "TFile.h"
#include "TH1.h"
#include "TString.h"

#include <iostream>

void PlotRMS() {

	std::vector<int> layer;

	layer.push_back(2);

	std::string InputFile = "../../run/output/shape_para/photon/NNinput_photon_50.000000GeV_eta_0.200000_0.250000_layer2_PCAbin1.root";


	std::string OutputFile = "../../run/output/shape_para/photon/NNinput_photon_50.000000GeV_eta_0.200000_0.250000_RMS.root";



	TFile *fin = TFile::Open(InputFile.c_str());

	TH1F *histo = (TH1F*)fin->Get("hDrEnergy");

	int nbins = histo->GetNbinsX();
	double nBinsR[nbins + 1];

	cout << " sizeof(nBinsR) = " << sizeof(nBinsR) << endl;



	for (int i = 1; i <= histo->GetNbinsX() + 1; i++) {

		nBinsR[i - 1] = histo->GetBinLowEdge(i);

	}

	for (int i = 0; i < histo->GetNbinsX(); i++) {

		std::cout << nBinsR[i] << std::endl;
	}

	TH1F *hRMS = new TH1F("hRMS", "hRMS", nbins, nBinsR);


	for (int i = 1; i < histo->GetNbinsX(); i++) {
		hRMS->SetBinContent(i, histo->GetBinError(i));

	}

	TFile *fout = new TFile(OutputFile.c_str(), "recreate");


	fout->cd();
	hRMS->Write();
	fout->Close();






}

