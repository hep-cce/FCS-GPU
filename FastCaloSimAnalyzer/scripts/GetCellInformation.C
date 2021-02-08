/*
  Copyright (C) 2002-2017 CERN for the benefit of the ATLAS collaboration
*/
#include "TFCSAnalyzerBase.h"
#include "TROOT.h"
#include "TFile.h"
#include "TH2.h"
#include "TString.h"

#include <iostream>

void GetCellInformation() {

	std::vector<int> layers;
	layers.push_back(1);
	layers.push_back(2);
	layers.push_back(3);
	// layers.push_back(12);




	// std::string file = "/eos/atlas/atlascerngroupdisk/proj-simul/InputSamplesProdsysProduction/mc16_13TeV.430400.ParticleGun_pid22_E1024_disj_eta_m5_m0_0_5_zv_0.deriv.NTUP_FCS.e6556_e5984_s3259_r10283_p3449/NTUP_FCS.13289183._000002.pool.root.1";

	// std::string file = "/eos/atlas/atlascerngroupdisk/proj-simul/InputSamplesProdsysProduction/mc16_13TeV.430404.ParticleGun_pid22_E1024_disj_eta_m25_m20_20_25_zv_0.deriv.NTUP_FCS.e6556_e5984_s3259_r10283_p3449/NTUP_FCS.13289193._000001.pool.root.1";

	std::string file = "/eos/atlas/atlascerngroupdisk/proj-simul/InputSamplesProdsysProduction/mc16_13TeV.431202.ParticleGun_pid22_E262144_disj_eta_m15_m10_10_15_zv_0.deriv.NTUP_FCS.e6556_e5984_s3259_r10283_p3449/NTUP_FCS.13289425._000001.pool.root.1";

	std::vector<float>* m_truthPx;
	std::vector<float>* m_truthPy;
	std::vector<float>* m_truthPz;
	std::vector<float>* m_truthE;

	std::vector<vector<float>>* m_TTC_mid_eta;
	std::vector<vector<float>>* m_TTC_mid_phi;
	FCS_matchedcellvector *m_cellVector;


	FCS_matchedcellvector *m_cellVector1;
	FCS_matchedcellvector *m_cellVector2;
	FCS_matchedcellvector *m_cellVector3;
	FCS_matchedcellvector *m_cellVector4;
	FCS_matchedcellvector *m_cellVector5;
	FCS_matchedcellvector *m_cellVector6;
	FCS_matchedcellvector *m_cellVector7;




	// TH1F* h_eta_good = new TH1F("h_eta_good", "h_eta_good", 2000, -.1, .1);
	// TH1F* h_truth_eta_good = new TH1F("h_truth_eta_good", "h_truth_eta_good", 10000, -.05, .05);

	// TH1F* h_eta_bad = new TH1F("h_eta_bad", "h_eta_bad", 10000, -.05, .05);
	// TH1F* h_truth_eta_bad = new TH1F("h_truth_eta_bad", "h_truth_eta_bad", 10000, -.05, .05);



	TChain *mychain = new TChain("FCS_ParametrizationInput");
	mychain->Add(file.c_str());


	m_truthPx         = nullptr;
	m_truthPy         = nullptr;
	m_truthPz         = nullptr;
	m_truthE          = nullptr;

	// m_cellVector = nullptr;
	m_cellVector1 = nullptr;
	m_cellVector2 = nullptr;
	m_cellVector3 = nullptr;
	m_cellVector4 = nullptr;
	m_cellVector5 = nullptr;
	m_cellVector6 = nullptr;
	m_cellVector7 = nullptr;


	m_TTC_mid_eta = nullptr;
	m_TTC_mid_phi = nullptr;


	// TString b_Sampling = Form("Sampling_%i", layer);
	mychain->SetBranchAddress("Sampling_1", &m_cellVector1);
	mychain->SetBranchAddress("Sampling_2", &m_cellVector2);
	mychain->SetBranchAddress("Sampling_3", &m_cellVector3);
	mychain->SetBranchAddress("Sampling_4", &m_cellVector4);
	mychain->SetBranchAddress("Sampling_5", &m_cellVector5);
	mychain->SetBranchAddress("Sampling_6", &m_cellVector6);
	mychain->SetBranchAddress("Sampling_7", &m_cellVector7);




	mychain->SetBranchAddress("newTTC_mid_eta", &m_TTC_mid_eta);
	mychain->SetBranchAddress("newTTC_mid_phi", &m_TTC_mid_phi);
	mychain->SetBranchAddress("TruthPx", &m_truthPx);
	mychain->SetBranchAddress("TruthPy", &m_truthPy);
	mychain->SetBranchAddress("TruthPz", &m_truthPz);
	mychain->SetBranchAddress("TruthE", &m_truthE);



	int nentries = mychain->GetEntries();

	for (int ientry = 0 ; ientry < nentries; ientry++) {
		mychain->GetEntry(ientry);

		unsigned int ncells1 = m_cellVector1->size();
		unsigned int ncells2 = m_cellVector2->size();
		unsigned int ncells3 = m_cellVector3->size();
		unsigned int ncells4 = m_cellVector4->size();
		unsigned int ncells5 = m_cellVector5->size();
		unsigned int ncells6 = m_cellVector6->size();
		unsigned int ncells7 = m_cellVector7->size();



		float sum_cell_energy1 = 0.;
		float sum_cell_energy2 = 0.;
		float sum_cell_energy3 = 0.;
		float sum_cell_energy4 = 0.;
		float sum_cell_energy5 = 0.;
		float sum_cell_energy6 = 0.;
		float sum_cell_energy7 = 0.;


		float TTC_eta = (*m_TTC_mid_eta).at(0).at(1);


		for (unsigned int icell = 0; icell < ncells1; icell++)
		{
			unsigned int nhits = m_cellVector1->m_vector.at(icell).hit.size();

			float cell_energy = m_cellVector1->m_vector.at(icell).cell.energy;
			sum_cell_energy1 += cell_energy;
		}

		for (unsigned int icell = 0; icell < ncells2; icell++)
		{
			unsigned int nhits = m_cellVector2->m_vector.at(icell).hit.size();

			float cell_energy = m_cellVector2->m_vector.at(icell).cell.energy;
			sum_cell_energy2 += cell_energy;
		}

		for (unsigned int icell = 0; icell < ncells3; icell++)
		{
			unsigned int nhits = m_cellVector3->m_vector.at(icell).hit.size();

			float cell_energy = m_cellVector3->m_vector.at(icell).cell.energy;
			sum_cell_energy3 += cell_energy;
		}


		for (unsigned int icell = 0; icell < ncells4; icell++)
		{
			unsigned int nhits = m_cellVector4->m_vector.at(icell).hit.size();

			float cell_energy = m_cellVector4->m_vector.at(icell).cell.energy;
			sum_cell_energy4 += cell_energy;
		}


		for (unsigned int icell = 0; icell < ncells5; icell++)
		{
			unsigned int nhits = m_cellVector5->m_vector.at(icell).hit.size();

			float cell_energy = m_cellVector5->m_vector.at(icell).cell.energy;
			sum_cell_energy5 += cell_energy;
		}

		for (unsigned int icell = 0; icell < ncells6; icell++)
		{
			unsigned int nhits = m_cellVector6->m_vector.at(icell).hit.size();

			float cell_energy = m_cellVector6->m_vector.at(icell).cell.energy;
			sum_cell_energy6 += cell_energy;
		}

		for (unsigned int icell = 0; icell < ncells7; icell++)
		{
			unsigned int nhits = m_cellVector7->m_vector.at(icell).hit.size();

			float cell_energy = m_cellVector7->m_vector.at(icell).cell.energy;
			sum_cell_energy7 += cell_energy;
		}

		if (sum_cell_energy1 == 0 and sum_cell_energy2 == 0 and sum_cell_energy3 == 0 and sum_cell_energy4 == 0 and sum_cell_energy5 == 0 and sum_cell_energy6 == 0 and sum_cell_energy7 == 0)
			std::cout << " Event number = " << ientry << ", TTC eta = " << TTC_eta << std::endl;


	}


	// for (int ilayer = 0; ilayer < layers.size(); ilayer++)
	// {
	// 	int layer = layers.at(ilayer);

	// 	std::cout << " Running on layer = " << layer << std::endl;

	// 	m_truthPx         = nullptr;
	// 	m_truthPy         = nullptr;
	// 	m_truthPz         = nullptr;
	// 	m_truthE          = nullptr;

	// 	m_cellVector = nullptr;
	// 	m_TTC_mid_eta = nullptr;
	// 	m_TTC_mid_phi = nullptr;


	// 	TString b_Sampling = Form("Sampling_%i", layer);
	// 	mychain->SetBranchAddress(b_Sampling, &m_cellVector);
	// 	mychain->SetBranchAddress("newTTC_mid_eta", &m_TTC_mid_eta);
	// 	mychain->SetBranchAddress("newTTC_mid_phi", &m_TTC_mid_phi);
	// 	mychain->SetBranchAddress("TruthPx", &m_truthPx);
	// 	mychain->SetBranchAddress("TruthPy", &m_truthPy);
	// 	mychain->SetBranchAddress("TruthPz", &m_truthPz);
	// 	mychain->SetBranchAddress("TruthE", &m_truthE);




	// 	int nentries = mychain->GetEntries();

	// 	for (int ientry = 0 ; ientry < nentries; ientry++) {
	// 		mychain->GetEntry(ientry);


	// 		float px = m_truthPx->at(0);
	// 		float py = m_truthPy->at(0);
	// 		float pz = m_truthPz->at(0);
	// 		float E = m_truthE->at(0);


	// 		TLorentzVector truthTLV;
	// 		truthTLV.SetPxPyPzE(px, py, pz, E);

	// 		float truth_eta = truthTLV.Eta();
	// 		float truth_phi = truthTLV.Phi();

	// 		float TTC_eta = (*m_TTC_mid_eta).at(0).at(layer);
	// 		float TTC_phi = (*m_TTC_mid_phi).at(0).at(layer);

	// 		// if (ientry != 1118 and ientry != 1382 and ientry != 1859 and ientry != 3608 and ientry != 5568 and ientry != 7047 and ientry != 8435 and ientry != 8750 and ientry != 1662 and ientry != 218 and ientry != 2028 and ientry != 0) {
	// 		// 	h_eta_good->Fill(TTC_eta);
	// 		// 	h_truth_eta_good->Fill(truth_eta);

	// 		// } else {
	// 		// 	h_eta_bad->Fill(TTC_eta);
	// 		// 	h_truth_eta_bad->Fill(truth_eta);
	// 		// }

	// 		// if (ientry != 1118) continue;

	// 		// TTC_r = (*m_TTC_mid_r).at(0).at(layer);

	// 		// std::cout << " Event number = " << ientry << ", TTC eta = " << TTC_eta << ", TTC phi = " << TTC_phi << std::endl;


	// 		unsigned int ncells = m_cellVector->size();

	// 		float sum_cell_energy = 0.;

	// 		for (unsigned int icell = 0; icell < ncells; icell++)
	// 		{
	// 			unsigned int nhits = m_cellVector->m_vector.at(icell).hit.size();

	// 			float cell_energy = m_cellVector->m_vector.at(icell).cell.energy;
	// 			sum_cell_energy += cell_energy;
	// 		}

	// 		if (sum_cell_energy == 0)
	// 			std::cout << " layer = " << layer << ", Event number = " << ientry << std::endl;


	// 	}

	// }


	// h_eta_bad->SetMarkerStyle(4);
	// h_eta_bad->SetMarkerColor(kRed);
	// h_eta_bad->SetMarkerSize(.5);
	// h_eta_good->SetMarkerStyle(6);


	// h_eta_good->SetMarkerSize(6);
	// h_eta_good->GetXaxis()->SetTitle("TTC eta");
	// h_eta_good->SetTitle("");
	// h_eta_good->GetXaxis()->SetLimits(-0.05, 0.05);




	// TCanvas* c1 =  new TCanvas();
	// c1->cd();
	// gStyle->SetOptStat(0);
	// h_eta_good->Draw("p");
	// h_eta_bad->Draw("p same");




}
