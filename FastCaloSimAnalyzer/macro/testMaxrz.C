/*
  Copyright (C) 2002-2017 CERN for the benefit of the ATLAS collaboration
*/
#include "TFCSAnalyzerBase.h"
#include "TROOT.h"
#include "TFile.h"
#include "TH2.h"
#include "TString.h"

#include <iostream>

void testMaxrz() {


	std::string file = "/eos/atlas/atlascerngroupdisk/proj-simul/InputSamplesProdsysProduction/mc16_13TeV.431202.ParticleGun_pid22_E262144_disj_eta_m15_m10_10_15_zv_0.deriv.NTUP_FCS.e6556_e5984_s3259_r10283_p3449/NTUP_FCS.13289425._000001.pool.root.1";


	TH1D* hrmax = new TH1D("hrmax", "hrmax", 30000, 0, 30000);
	TH1D* hzmax = new TH1D("hzmax", "hzmax", 1000, -500, 500);



	FCS_matchedcellvector *m_cellVector;
	std::vector<vector<float>>* m_TTC_mid_eta;
	std::vector<vector<float>>* m_TTC_mid_phi;
	std::vector<vector<float>>* m_TTC_mid_r;
	std::vector<vector<float>>* m_TTC_mid_z;


	TChain *mychain = new TChain("FCS_ParametrizationInput");
	mychain->Add(file.c_str());


	m_cellVector   = nullptr;
	m_TTC_mid_eta = nullptr;
	m_TTC_mid_phi = nullptr;
	m_TTC_mid_r = nullptr;
	m_TTC_mid_z = nullptr;





	mychain->SetBranchAddress("newTTC_mid_eta", &m_TTC_mid_eta);
	mychain->SetBranchAddress("newTTC_mid_phi", &m_TTC_mid_phi);
	mychain->SetBranchAddress("newTTC_mid_r", &m_TTC_mid_r);
	mychain->SetBranchAddress("newTTC_mid_z", &m_TTC_mid_z);



	int nentries = mychain->GetEntries();
	// nentries = 5;

	for (int ilayer = 0; ilayer < 24; ilayer++)
	{
		if (ilayer != 2) continue;

		float max_extrapol_r = -1;
		float max_extrapol_z = -1;

		int layer = ilayer;
		TString b_Sampling = Form("Sampling_%i", layer);

		mychain->SetBranchAddress(b_Sampling, &m_cellVector);

		for (int ievent = 0; ievent < nentries; ievent++)
		{
			// if (ievent % 1000 == 0)
			// 	std::cout << " Event: " << ievent << std::endl;


			mychain->GetEntry(ievent);

			float TTC_eta = (*m_TTC_mid_eta).at(0).at(layer);
			float TTC_phi = (*m_TTC_mid_phi).at(0).at(layer);
			float TTC_r = (*m_TTC_mid_r).at(0).at(layer);
			float TTC_z = (*m_TTC_mid_z).at(0).at(layer);
			unsigned int ncells = m_cellVector->size();

			// cout << " TTC r = " << TTC_r << endl;
			// cout << " TTC z = " << TTC_z << endl;

			if (TTC_r > max_extrapol_r) max_extrapol_r = TTC_r;
			if (TTC_z > max_extrapol_z ) max_extrapol_z = TTC_z;

			TH1D* hr = new TH1D("hr", "hr", 300000, 0, 30000);

			TH1D* hz = new TH1D("hz", "hz", 10000, -5000, 5000);


			for (unsigned int icell = 0; icell < ncells; icell++)
			{
				unsigned int nhits = m_cellVector->m_vector.at(icell).hit.size();

				for (unsigned int ihit = 0; ihit < nhits;  ihit++)
				{
					float energy = m_cellVector->m_vector.at(icell).hit.at(ihit).hit_energy;
					float hit_x = m_cellVector->m_vector.at(icell).hit.at(ihit).hit_x;
					float hit_y = m_cellVector->m_vector.at(icell).hit.at(ihit).hit_y;
					float hit_z = m_cellVector->m_vector.at(icell).hit.at(ihit).hit_z;

					float hit_r = TMath::Sqrt(hit_x * hit_x + hit_y * hit_y);

					// std::cout << " energy = " << energy << std::endl;
					// std::cout << " hit z  = " << hit_z << std::endl;

					hr->Fill(hit_r, energy);
					hz->Fill(hit_z, energy);

				}
			}
			float rmax = hr->GetBinCenter(hr->GetMaximumBin());
			float zmax = hz->GetBinCenter(hz->GetMaximumBin());

			// cout << " rmax = " << rmax << endl;
			// cout << " zmax = " << zmax << endl;

			hr->Reset();
			hz->Reset();
			delete hr;
			delete hz;

			hrmax->Fill(rmax);
			hzmax->Fill(zmax);

		}

		TCanvas * c1 = new TCanvas("c1", "c1", 0, 0, 800, 500);
		c1->cd();
		hrmax->Draw("e");
		TCanvas * c2 = new TCanvas("c2", "c2", 0, 0, 800, 500);
		c2->cd();
		hzmax->Draw("e");

		// std::cout << "Layer = " << layer << "; max r = " << max_extrapol_r << "; z = " << max_extrapol_z << std::endl;

	}
}
