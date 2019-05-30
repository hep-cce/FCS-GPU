/*
  Copyright (C) 2002-2017 CERN for the benefit of the ATLAS collaboration
*/
#include "TFCSAnalyzerBase.h"
#include "TROOT.h"
#include "TFile.h"
#include "TH2.h"
#include "TString.h"

#include <iostream>

void GetLayerPosition() {

	// std::vector<int> layers;
	// layers.push_back(1);
	// layers.push_back(2);
	// layers.push_back(3);
	// // layers.push_back(12);




	// std::string file = "/eos/atlas/atlascerngroupdisk/proj-simul/InputSamplesProdsysProduction/mc16_13TeV.430400.ParticleGun_pid22_E1024_disj_eta_m5_m0_0_5_zv_0.deriv.NTUP_FCS.e6556_e5984_s3259_r10283_p3449/NTUP_FCS.13289183._000002.pool.root.1";

	std::string file = "/eos/atlas/atlascerngroupdisk/proj-simul/InputSamplesProdsysProduction/mc16_13TeV.430404.ParticleGun_pid22_E1024_disj_eta_m25_m20_20_25_zv_0.deriv.NTUP_FCS.e6556_e5984_s3259_r10283_p3449/NTUP_FCS.13289193._000001.pool.root.1";

	//std::string file = "/eos/atlas/atlascerngroupdisk/proj-simul/InputSamplesProdsysProduction/mc16_13TeV.431202.ParticleGun_pid22_E262144_disj_eta_m15_m10_10_15_zv_0.deriv.NTUP_FCS.e6556_e5984_s3259_r10283_p3449/NTUP_FCS.13289425._000001.pool.root.1";


	std::vector<vector<float>>* m_TTC_mid_eta;
	std::vector<vector<float>>* m_TTC_mid_phi;
	std::vector<vector<float>>* m_TTC_mid_r;
	std::vector<vector<float>>* m_TTC_mid_z;



	// FCS_matchedcellvector *m_cellVector;






	// TH1F* h_eta_good = new TH1F("h_eta_good", "h_eta_good", 2000, -.1, .1);
	// TH1F* h_truth_eta_good = new TH1F("h_truth_eta_good", "h_truth_eta_good", 10000, -.05, .05);

	// TH1F* h_eta_bad = new TH1F("h_eta_bad", "h_eta_bad", 10000, -.05, .05);
	// TH1F* h_truth_eta_bad = new TH1F("h_truth_eta_bad", "h_truth_eta_bad", 10000, -.05, .05);



	TChain *mychain = new TChain("FCS_ParametrizationInput");
	mychain->Add(file.c_str());


	m_TTC_mid_eta = nullptr;
	m_TTC_mid_phi = nullptr;
	m_TTC_mid_r = nullptr;
	m_TTC_mid_z = nullptr;






	mychain->SetBranchAddress("newTTC_mid_eta", &m_TTC_mid_eta);
	mychain->SetBranchAddress("newTTC_mid_phi", &m_TTC_mid_phi);
	mychain->SetBranchAddress("newTTC_mid_r", &m_TTC_mid_r);
	mychain->SetBranchAddress("newTTC_mid_z", &m_TTC_mid_z);






	int nentries = mychain->GetEntries();


	mychain->GetEntry(0);


	for (int ilayer = 0; ilayer < 24; ilayer++)
	{

		int layer = ilayer;

		float TTC_eta = (*m_TTC_mid_eta).at(0).at(layer);
		float TTC_phi = (*m_TTC_mid_phi).at(0).at(layer);
		float TTC_r = (*m_TTC_mid_r).at(0).at(layer);
		float TTC_z = (*m_TTC_mid_z).at(0).at(layer);


		std::cout << "Layer = " << layer << "; r = " << TTC_r << "; z = " << TTC_z << std::endl;



	}


}
