/*
  Copyright (C) 2002-2017 CERN for the benefit of the ATLAS collaboration
*/

#include "TROOT.h"
#include "TFile.h"
#include "TH2.h"
#include "TString.h"

#include <iostream>

void CreateHitsGrid(std::string particle) {


	std::vector<int> layer;
// std::vector<int> pca;

	layer.push_back(0);
	layer.push_back(1);
	layer.push_back(2);
	layer.push_back(3);
	layer.push_back(12);
	//layer.push_back(13);
	//layer.push_back(14);

	std::string outFile = "../../run/output/shape_para/" + particle + "/nHits_" + particle + "_50.000000GeV_eta_0.200000_0.250000.csv";

	std::cout << " csv file = " << outFile.c_str() << endl;

	ofstream mycsv;
	mycsv.open(outFile.c_str());

	if (mycsv.is_open())
	{

		mycsv << "CaloLayer, PCAbin, nHits, nEvents" << endl;


		for (int i = 0; i < layer.size(); ++i)
		{
			for (int j = 1; j < 6; ++j)
			{
				std::string file = "../../run/output/shape_para/" + particle + "/HitsAlphaDr_" + particle + "_50.000000GeV_eta_0.200000_0.250000_layer" + std::to_string(layer.at(i)) + "_PCAbin" + std::to_string(j) + ".root";



				TFile * f = TFile::Open(file.c_str());
				TTree* t = (TTree*)f->Get("hitsAlphaDr");

				int m_eventNumber = -1;
				t->SetBranchAddress("eventNumber", &m_eventNumber);

				int nevents = 2;

				for (int i = 0; i < t->GetEntries() - 1 ; i++) {

					t->GetEntry(i);
					int current_evt = m_eventNumber;
					t->GetEntry(i + 1);
					int next_evt = m_eventNumber;

					if (current_evt != next_evt) nevents++;

				}

				mycsv << layer.at(i) << ", " << j << ", " << t->GetEntries() << ", " << nevents << endl;

				f->Close();
			}
		}

		mycsv.close();
	}

}