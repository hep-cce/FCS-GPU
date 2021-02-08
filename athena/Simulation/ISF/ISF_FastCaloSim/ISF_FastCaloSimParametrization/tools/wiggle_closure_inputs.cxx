/*
  Copyright (C) 2002-2017 CERN for the benefit of the ATLAS collaboration
*/

#include "FCS_Cell.h"
#include "../ISF_FastCaloSimParametrization/MeanAndRMS.h"
#include "Identifier/Identifier.h"
#include "CaloDetDescr/CaloDetDescrElement.h"
#include "CaloGeometryFromFile.h"
#include "CaloHitAna.h"
#include "../ISF_FastCaloSimParametrization/CaloGeometry.h"
#include "TFile.h"
#include "TTree.h"
#include "TString.h"
#include <string>
#include <sstream>
#include <iostream>
#include "TSystem.h"
#include "TString.h"
#include "TFile.h"
#include <stdlib.h>
#include "TLorentzVector.h"
#include "TH1.h"
#include "TH2.h"
#include "TH1F.h"
#include "TH2F.h"
#include <vector>
#include "TCanvas.h"
//#include "MakeBins.C"
using namespace std;

void wiggle_closure_inputs(TString sampling="Sampling_0"){

  CaloGeometryFromFile* geo=new CaloGeometryFromFile();
  geo->LoadGeometryFromFile("/afs/cern.ch/atlas/groups/Simulation/FastCaloSim/ATLAS-GEO-20-00-01.root","ATLAS-GEO-20-00-01");
  geo->LoadFCalGeometryFromFiles("FCal1-electrodes.sorted.HV.09Nov2007.dat","FCal2-electrodes.sorted.HV.April2011.dat","FCal3-electrodes.sorted.HV.09Nov2007.dat");
  
  TFile *inputFile = TFile::Open("/eos/atlas/user/s/schaarsc/FCS/user.fladias.428137.FastCalo_pid11_E65536_etam35_35_zv_m100.e4001_s2864_r7736.w0_162706_matched_output.root/user.fladias.8834798._000001.matched_output.root");
 
	// TFile *inputFile = TFile::Open("root://eosatlas//eos/atlas/user/z/zhubacek/FastCaloSim/LArShift020715/ISF_HitAnalysis6_evgen_calo__211_E50000_50000_eta20_25_Evts0-5500_vz_0_origin_calo.merged.pool.root");
  //  TFile *inputFile = TFile::Open("root://eosatlas//eos/atlas/user/z/zhubacek/FastCaloSim/NTUP_110216/ISF_HitAnalysis_Zach1_merged.root");

  TTree *inputTree = ( TTree* ) inputFile->Get( "FCS_ParametrizationInput" );

  FCS_matchedcellvector *vec=0; //this need to be set to 0!
  inputTree->SetBranchAddress(sampling,&vec);

  //histograms definition
  TH1F *eff_tot_phi = new TH1F("eff_tot_phi_"+sampling, "eff_tot_phi_"+sampling, 50, -1, 1);
  TH1F *eff_corr_phi = new TH1F("eff_corr_phi_"+sampling, "eff_corr_phi_"+sampling, 50, -1, 1);
  TH1F *eff_tot_count_phi = new TH1F("eff_tot_count_phi_"+sampling, "eff_tot_count_phi_"+sampling, 50, -1, 1);
  TH1F *eff_corr_count_phi = new TH1F("eff_corr_count_phi_"+sampling, "eff_corr_count_phi_"+sampling, 50, -1, 1);
  eff_corr_phi->Sumw2();
  eff_tot_phi->Sumw2();
  eff_corr_count_phi->Sumw2();
  eff_tot_count_phi->Sumw2();


  std::cout << "Sampling is " << sampling << std::endl;

  Int_t nEvt = inputTree->GetEntries();
  std::cout << "nEvt " << nEvt << std::endl;
  
  for (Int_t ientry=0; ientry<nEvt; ientry++){
    
    inputTree->GetEntry(ientry);   
    if(ientry%100 == 0)  
      std::cout << "Processing event # " << ientry << std::endl;

    double weight = 1.0;
    
    //loop over cells in sampling
    for (UInt_t j=0; j<(*vec).size(); j++){
      
      Float_t cell_E = 0.0;
      cell_E = ((FCS_matchedcell)(*vec)[j]).cell.energy;
      Long64_t cell_ID = 0;
      cell_ID = ((FCS_matchedcell)(*vec)[j]).cell.cell_identifier;

      //now I use the geomery lookup tool to get the cell eta/phi
      const CaloGeoDetDescrElement* cell;
      Identifier cellid(cell_ID);
      cell=geo->getDDE(cellid); //This is working also for the FCal
      
      //loop over hits in the cell
      for (unsigned int ihit=0; ihit<((FCS_matchedcell)(*vec)[j]).hit.size(); ihit++){
	
	//now I check what is the hit position
	float x = ((FCS_matchedcell)(*vec)[j]).hit[ihit].hit_x;
	float y = ((FCS_matchedcell)(*vec)[j]).hit[ihit].hit_y;
	float z = ((FCS_matchedcell)(*vec)[j]).hit[ihit].hit_z;
	float t = ((FCS_matchedcell)(*vec)[j]).hit[ihit].hit_time;
	int hitSampling =((FCS_matchedcell)(*vec)[j]).hit[ihit].sampling;
	
	TLorentzVector *hitVec = new TLorentzVector(x, y, z, t);
	const CaloGeoDetDescrElement* foundCell;
	if(hitSampling<21)foundCell=geo->getDDE(hitSampling,hitVec->Eta(),hitVec->Phi());
	else if(hitSampling<24)foundCell=geo->getFCalDDE(hitSampling,x,y,z);
	else {
		cout << endl << "Warning: Found hit with sampling > 23 !!!!!!!!!!!!!!" << endl << endl;
		foundCell =0;
	}
	
	if (foundCell){// && cell){
	  if(hitSampling<21){
		  eff_tot_phi->Fill(((2*(hitVec->Phi()-foundCell->phi()))/foundCell->dphi()),weight);	  
		  eff_tot_count_phi->Fill(((2*(hitVec->Phi()-foundCell->phi()))/foundCell->dphi()));
		  
		  if (foundCell->identify() == cell->identify()){
		    
		    eff_corr_phi->Fill(((2*(hitVec->Phi()-foundCell->phi()))/foundCell->dphi()),weight);
		    eff_corr_count_phi->Fill(((2*(hitVec->Phi()-foundCell->phi()))/foundCell->dphi()));
		  }
		}
		else if(hitSampling<24){
		  eff_tot_phi->Fill(((2*(x-foundCell->x()))/foundCell->dx()),weight);	  
		  eff_tot_count_phi->Fill(((2*(x-foundCell->x()))/foundCell->dx()));
		  
		  if (foundCell->identify() == cell->identify()){
		    
		    eff_corr_phi->Fill(((2*(x-foundCell->x()))/foundCell->dx()),weight);
		    eff_corr_count_phi->Fill(((2*(x-foundCell->x()))/foundCell->dx()));
		  }
		}
	  
	} //end if cell ok
      } //end loop over hits
    } //end loop on cells   
  } //end loop on events
  
  
  /*
  for (Int_t bin1=1; bin1<301; bin1++){
    for (Int_t bin2=1; bin2<301; bin2++){
      if (eff_tot_count_phi->GetBinContent(bin1) < 2){
	eff_tot_phi->SetBinContent(bin1,0.0);
	eff_corr_phi->SetBinContent(bin1,0.0);
      }
      if (eff_corr_count_phi->GetBinContent(bin1) < 2){
	eff_tot_phi->SetBinContent(bin1,0.0);
	eff_corr_phi->SetBinContent(bin1,0.0);
      }
      
      }}*/
  
  TCanvas *c1 = new TCanvas();
  eff_tot_phi->Draw();
  TCanvas *c2 = new TCanvas();
  eff_corr_phi->Draw();

  TCanvas *c5 = new TCanvas();
  TH1F *eff_phi;
  eff_phi = (TH1F*)eff_corr_phi->Clone("eff_phi_"+sampling);
  eff_phi->Divide(eff_tot_phi);
  eff_phi->Scale(1.0/eff_phi->Integral());
  cout << eff_phi->Integral() << endl;
  eff_phi->SetTitle("Efficiency of cell assigment");
  eff_phi->GetXaxis()->SetTitle("2*(#phi_{hit}-#phi_{assigCell})/dphi_{assigCell}");
  eff_phi->GetYaxis()->SetTitle("Efficiency of cell assignment");
  eff_phi->Draw();
  
  double eff_int[50]; 
  eff_int[0] = eff_phi->GetBinContent(1);
  for (int i=1; i<50; i++){
    eff_int[i] = eff_int[i-1] + eff_phi->GetBinContent(i+1);
  }
  
  for (int j=0; j<50; j++){
    cout << eff_int[j] << endl;
  }

  TFile f("wiggle_efficiency_"+sampling+".root", "recreate");
  eff_phi->Write();
  //eff_int->Write();
  f.Close();
  



}
