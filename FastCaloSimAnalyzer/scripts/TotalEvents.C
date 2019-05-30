/*
  Copyright (C) 2002-2017 CERN for the benefit of the ATLAS collaboration
*/

#include "TChain.h"
#include "TFile.h"
#include "TTree.h"

#include <iostream>
#include <stdlib.h>


using namespace std;

void TotalEvents() {

	vector < string > input;

	input.push_back("root://eosatlas//eos/atlas/user/s/schaarsc/FCS/user.fladias.428137.FastCalo_pid11_E65536_etam35_35_zv_m100.e4001_s2865_r7736.w0_162706_matched_output.root/user.fladias.8834800._000001.matched_output.root");
	input.push_back("root://eosatlas//eos/atlas/user/s/schaarsc/FCS/user.fladias.428137.FastCalo_pid11_E65536_etam35_35_zv_m100.e4001_s2865_r7736.w0_162706_matched_output.root/user.fladias.8834800._000002.matched_output.root");
	input.push_back("root://eosatlas//eos/atlas/user/s/schaarsc/FCS/user.fladias.428137.FastCalo_pid11_E65536_etam35_35_zv_m100.e4001_s2865_r7736.w0_162706_matched_output.root/user.fladias.8834800._000003.matched_output.root");
	input.push_back("root://eosatlas//eos/atlas/user/s/schaarsc/FCS/user.fladias.428137.FastCalo_pid11_E65536_etam35_35_zv_m100.e4001_s2865_r7736.w0_162706_matched_output.root/user.fladias.8834800._000004.matched_output.root");
	input.push_back("root://eosatlas//eos/atlas/user/s/schaarsc/FCS/user.fladias.428137.FastCalo_pid11_E65536_etam35_35_zv_m100.e4001_s2865_r7736.w0_162706_matched_output.root/user.fladias.8834800._000005.matched_output.root");
	input.push_back("root://eosatlas//eos/atlas/user/s/schaarsc/FCS/user.fladias.428137.FastCalo_pid11_E65536_etam35_35_zv_m100.e4001_s2865_r7736.w0_162706_matched_output.root/user.fladias.8834800._000006.matched_output.root");
	input.push_back("root://eosatlas//eos/atlas/user/s/schaarsc/FCS/user.fladias.428137.FastCalo_pid11_E65536_etam35_35_zv_m100.e4001_s2865_r7736.w0_162706_matched_output.root/user.fladias.8834800._000007.matched_output.root");
	input.push_back("root://eosatlas//eos/atlas/user/s/schaarsc/FCS/user.fladias.428137.FastCalo_pid11_E65536_etam35_35_zv_m100.e4001_s2865_r7736.w0_162706_matched_output.root/user.fladias.8834800._000008.matched_output.root");
	input.push_back("root://eosatlas//eos/atlas/user/s/schaarsc/FCS/user.fladias.428137.FastCalo_pid11_E65536_etam35_35_zv_m100.e4001_s2865_r7736.w0_162706_matched_output.root/user.fladias.8834800._000009.matched_output.root");
	input.push_back("root://eosatlas//eos/atlas/user/s/schaarsc/FCS/user.fladias.428137.FastCalo_pid11_E65536_etam35_35_zv_m100.e4001_s2865_r7736.w0_162706_matched_output.root/user.fladias.8834800._000010.matched_output.root");
	input.push_back("root://eosatlas//eos/atlas/user/s/schaarsc/FCS/user.fladias.428137.FastCalo_pid11_E65536_etam35_35_zv_m100.e4001_s2865_r7736.w0_162706_matched_output.root/user.fladias.8834800._000011.matched_output.root");
	input.push_back("root://eosatlas//eos/atlas/user/s/schaarsc/FCS/user.fladias.428137.FastCalo_pid11_E65536_etam35_35_zv_m100.e4001_s2865_r7736.w0_162706_matched_output.root/user.fladias.8834800._000012.matched_output.root");
	input.push_back("root://eosatlas//eos/atlas/user/s/schaarsc/FCS/user.fladias.428137.FastCalo_pid11_E65536_etam35_35_zv_m100.e4001_s2865_r7736.w0_162706_matched_output.root/user.fladias.8834800._000013.matched_output.root");
	input.push_back("root://eosatlas//eos/atlas/user/s/schaarsc/FCS/user.fladias.428137.FastCalo_pid11_E65536_etam35_35_zv_m100.e4001_s2865_r7736.w0_162706_matched_output.root/user.fladias.8834800._000014.matched_output.root");
	input.push_back("root://eosatlas//eos/atlas/user/s/schaarsc/FCS/user.fladias.428137.FastCalo_pid11_E65536_etam35_35_zv_m100.e4001_s2865_r7736.w0_162706_matched_output.root/user.fladias.8834800._000015.matched_output.root");
	input.push_back("root://eosatlas//eos/atlas/user/s/schaarsc/FCS/user.fladias.428137.FastCalo_pid11_E65536_etam35_35_zv_m100.e4001_s2865_r7736.w0_162706_matched_output.root/user.fladias.8834800._000016.matched_output.root");
	input.push_back("root://eosatlas//eos/atlas/user/s/schaarsc/FCS/user.fladias.428137.FastCalo_pid11_E65536_etam35_35_zv_m100.e4001_s2865_r7736.w0_162706_matched_output.root/user.fladias.8834800._000017.matched_output.root");
	input.push_back("root://eosatlas//eos/atlas/user/s/schaarsc/FCS/user.fladias.428137.FastCalo_pid11_E65536_etam35_35_zv_m100.e4001_s2865_r7736.w0_162706_matched_output.root/user.fladias.8834800._000018.matched_output.root");
	input.push_back("root://eosatlas//eos/atlas/user/s/schaarsc/FCS/user.fladias.428137.FastCalo_pid11_E65536_etam35_35_zv_m100.e4001_s2865_r7736.w0_162706_matched_output.root/user.fladias.8834800._000019.matched_output.root");
	input.push_back("root://eosatlas//eos/atlas/user/s/schaarsc/FCS/user.fladias.428137.FastCalo_pid11_E65536_etam35_35_zv_m100.e4001_s2865_r7736.w0_162706_matched_output.root/user.fladias.8834800._000020.matched_output.root");
	input.push_back("root://eosatlas//eos/atlas/user/s/schaarsc/FCS/user.fladias.428137.FastCalo_pid11_E65536_etam35_35_zv_m100.e4001_s2865_r7736.w0_162706_matched_output.root/user.fladias.8834800._000021.matched_output.root");


	TChain *mychain = new TChain("FCS_ParametrizationInput");

	for (auto i : input)
	{
		mychain->Add(i.c_str());
	}


	cout << " * Prepare to run with entries = " << mychain->GetEntries() << endl;

}