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
#include "TRandom3.h"
#include "TRandom.h"
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
#include "TProfile2D.h"
#include "TVector2.h"
#include <vector>
#include "TCanvas.h"
#include "TSystem.h"
#include "TMath.h"
#include <map>

using namespace std;

void wiggleClosureAndComparison(TString sampling="Sampling_0"){
  
  //load geometry finder
  CaloGeometryFromFile* geo=new CaloGeometryFromFile();
  geo->LoadGeometryFromFile("/afs/cern.ch/atlas/groups/Simulation/FastCaloSim/ATLAS-GEO-20-00-01.root","ATLAS-GEO-20-00-01");
  geo->LoadFCalGeometryFromFiles("FCal1-electrodes.sorted.HV.09Nov2007.dat","FCal2-electrodes.sorted.HV.April2011.dat","FCal3-electrodes.sorted.HV.09Nov2007.dat");
  
  //TFile *originalFile = TFile::Open("root://eosatlas//eos/atlas/user/z/zhubacek/FastCaloSim/LArShift020715/ISF_HitAnalysis6_evgen_calo__211_E50000_50000_eta20_25_Evts0-5500_vz_0_origin_calo.merged.pool.root");
  TFile *originalFile = TFile::Open("/eos/atlas/user/s/schaarsc/FCS/user.fladias.428137.FastCalo_pid11_E65536_etam35_35_zv_m100.e4001_s2864_r7736.w0_162706_matched_output.root/user.fladias.8834798._000001.matched_output.root");
  
  TFile *PCAfile = TFile::Open("firstPCA_pions_flavia.root");
  
  TTree *inputTree = ( TTree* ) originalFile->Get( "FCS_ParametrizationInput" );
  FCS_matchedcellvector *vec=0; //this need to be set to 0!
  inputTree->SetBranchAddress(sampling,&vec);
  
  TTree *PCAtree = ( TTree* ) PCAfile->Get( "firstPCA" );
  int PCAbin = -99;
  PCAtree->SetBranchAddress("bin",&PCAbin);  
  
  std::vector<FCS_truth> *truth=0; //does this also need to be zero?
  inputTree->SetBranchAddress("TruthCollection",&truth);
  
  //Now instead of loading the file, I make the closure in the same job
  /* TFile *closureFile = TFile::Open("wiggle_ClosureOutput_"+sampling+".root");
  TTree *closureTree = ( TTree* ) closureFile->Get( "FCS_ClosureOutput" );
  vector<FCS_cell> *vect=0; //this need to be set to 0!
  closureTree->SetBranchAddress(sampling,&vect);
  */


  /////////////////////////////////
  //     Start of wiggle def     //
  /////////////////////////////////

  //efficiency inputs from file  
  TFile *effFile = TFile::Open("wiggle_efficiency_"+sampling+".root");
  TH1F *eff_phi = ( TH1F* ) effFile->Get( "eff_phi_"+sampling );

  //have to change to get the wiggle derivative instead
  double eff_int[50]; 
  eff_int[0] = eff_phi->GetBinContent(1);
  for (int i=1; i<50; i++){
    eff_int[i] = eff_int[i-1] + eff_phi->GetBinContent(i+1);
  }
  
  for (int j=0; j<50; j++){
    cout << eff_int[j] << endl;
  }
  
  /////////////////////////////////
  //      End of wiggle def      //
  /////////////////////////////////

 
  //Histogram definitions
  
  TProfile2D *dEta_dPhi[11];
  TH2F *dEta_dPhi_N[11];
  TH2F *dEta_dPhi_res[11];
  TH2F *dEta_dPhi_N_res[11];
  
  TProfile2D *dEta_dPhi_avg_reco[11];
  TProfile2D *dEta_dPhi_avg_g4[11];
  TProfile2D *dEta_dPhi_FCS_orig[11];
  TProfile2D *dEta_dPhi_FCS_matched[11];
  TProfile2D *dEta_dPhi_FCS_orig_overAvg[11];
  TProfile2D *dEta_dPhi_FCS_matched_overAvg[11];
  TH2F *dEta_dPhi_FCS_orig_overAvg_N[11];
  TH2F *dEta_dPhi_FCS_matched_overAvg_N[11];
  TProfile2D *subtraction[11];

  //10 is inclusive
  //0-9 is for each of the PCA bins
  
  int nbins_eta;
  int nbins_phi;
  
  if (sampling == "Sampling_1" || sampling == "Sampling_2" || sampling == "Sampling_3"){
    //cout << "Sampling_2" << endl;
    //    nbins_eta = 120;
    // nbins_phi = 120;
    nbins_eta = 120;
    nbins_phi = 120;
  }
  else if(sampling == "Sampling_12" || sampling == "Sampling_13"){
    nbins_eta = 12;
    nbins_phi = 12;
  }
  else if(sampling == "Sampling_0"){
    nbins_eta = 60;
    nbins_phi = 60;
  }

  else if(sampling == "Sampling_14"){
    nbins_eta = 10;
    nbins_phi = 10;
  } 

  for (int i=0; i<11; i++){
    TString layerName=sampling+"_Layer_";
    layerName+=i;
    
    dEta_dPhi[i] = new TProfile2D("closure_dEtadPhi_Layer_"+layerName, "closure_dEtadPhi_Layer_"+layerName, nbins_eta, -0.5, 0.5, nbins_phi, -0.5, 0.5);
    dEta_dPhi_N[i] = new TH2F("closure_dEtadPhi_N_Layer_"+layerName, "closure_dEtadPhi_N_Layer_"+layerName, nbins_eta, -0.5, 0.5, nbins_phi, -0.5, 0.5);
    
    dEta_dPhi_res[i] = new TH2F("closure_dEtadPhi_res_Layer_"+layerName, "closure_dEtadPhi_res_Layer_"+layerName, nbins_eta, -0.5, 0.5, nbins_phi, -0.5, 0.5);
    dEta_dPhi_N_res[i] = new TH2F("closure_dEtadPhi_N_res_Layer_"+layerName, "closure_dEtadPhi_N_res_Layer_"+layerName, nbins_eta, -0.5, 0.5, nbins_phi, -0.5, 0.5);
    
    dEta_dPhi_avg_reco[i] = new TProfile2D("dEta_dPhi_avg_reco_Layer_"+layerName, "dEta_dPhi_avg_reco_Layer_"+layerName, nbins_eta, -0.5, 0.5, nbins_phi, -0.5, 0.5);
    dEta_dPhi_avg_g4[i] = new TProfile2D("dEta_dPhi_avg_g4_Layer_"+layerName, "dEta_dPhi_avg_g4_Layer_"+layerName, nbins_eta, -0.5, 0.5, nbins_phi, -0.5, 0.5);
    dEta_dPhi_FCS_orig[i] = new TProfile2D(" dEta_dPhi_FCS_orig_Layer_"+layerName, " dEta_dPhi_FCS_orig_Layer_"+layerName, nbins_eta, -0.5, 0.5, nbins_phi, -0.5, 0.5);
    dEta_dPhi_FCS_matched[i] = new TProfile2D("dEta_dPhi_FCS_matched_Layer_"+layerName, "dEta_dPhi_FCS_matched_Layer_"+layerName, nbins_eta, -0.5, 0.5, nbins_phi, -0.5, 0.5);

    dEta_dPhi_FCS_orig_overAvg[i] = new TProfile2D("dEta_dPhi_FCS_orig_overAvg_Layer_"+layerName, "dEta_dPhi_FCS_orig_overAvg_reco_Layer_"+layerName, nbins_eta, -0.5, 0.5, nbins_phi, -0.5, 0.5);
    dEta_dPhi_FCS_matched_overAvg[i] = new TProfile2D("dEta_dPhi_FCS_matched_overAvg_Layer_"+layerName, "dEta_dPhi_FCS_matched_overAvg_Layer_"+layerName, nbins_eta, -0.5, 0.5, nbins_phi, -0.5, 0.5);

    dEta_dPhi_FCS_orig_overAvg_N[i] = new TH2F("dEta_dPhi_FCS_orig_overAvg_N_Layer_"+layerName, "dEta_dPhi_FCS_orig_overAvg_N_reco_Layer_"+layerName, nbins_eta, -0.5, 0.5, nbins_phi, -0.5, 0.5);
    dEta_dPhi_FCS_matched_overAvg_N[i] = new TH2F("dEta_dPhi_FCS_matched_overAvg_N_Layer_"+layerName, "dEta_dPhi_FCS_matched_overAvg_N_Layer_"+layerName, nbins_eta, -0.5, 0.5, nbins_phi, -0.5, 0.5);

  }
  //end of histogram definitions

  //define maps to be used in analysis
  std::map<Long64_t, double> Eoriginal;
  std::map<Long64_t, double> EoriginalG4;  
  std::map<Long64_t, double> EoriginalReco;
  std::map<Long64_t, double> Eclosure;
  //end of maps definition

  std::cout << "Sampling is " << sampling << std::endl;

  Int_t nEvt = inputTree->GetEntries();
  std::cout << "nEvt " << nEvt << std::endl;

  Int_t nEvtPCA = PCAtree->GetEntries();
  std::cout << "nEvt3 " << nEvtPCA << std::endl;

  //start event loop
  for (Int_t ientry=0; ientry<nEvt; ientry++){
    
    inputTree->GetEntry(ientry);   
    if(ientry%100 == 0)  
      std::cout << "Processing event # " << ientry << std::endl;

    PCAtree->GetEntry(ientry);
    //cout << "PCAbin " << PCAbin << endl;
    //cout << "PCAevtN " << PCAevtN << endl;
    //cout << endl;

    //clean the maps before the start of the event
    Eoriginal.clear();
    EoriginalG4.clear();
    EoriginalReco.clear();
    Eclosure.clear();
 
    //variables to store, for this event
    double sum_energy_hit = 0.0;
    double sum_energy_g4hit = 0.0;
    double trEta = 0.0;
    double trPhi = 0.0;
    //cout << "TTC_entrance_eta " << (*truth)[0].TTC_entrance_eta.size() << " " << (*truth)[0].TTC_entrance_eta[0] << endl; 
    //cout << "TTC_entrance_phi " << (*truth)[0].TTC_entrance_phi.size() << " " << (*truth)[0].TTC_entrance_phi[0] << endl; 
    //cout << endl;
    trEta = (*truth)[0].TTC_entrance_eta[0];
    trPhi = (*truth)[0].TTC_entrance_phi[0];
    
    //loop over cells in sampling to sum its energy on the layer
    for (UInt_t j=0; j<(*vec).size(); j++){    
      sum_energy_hit = 0.0;
      sum_energy_g4hit = 0.0;

      //loop over hits in each cell to find the sum of the hit energies in the layer
      for (int ihit=0; ihit<((FCS_matchedcell)(*vec)[j]).hit.size(); ihit++){
        sum_energy_hit = sum_energy_hit + ((FCS_matchedcell)(*vec)[j]).hit[ihit].hit_energy;
      }

      for (int ihit=0; ihit<((FCS_matchedcell)(*vec)[j]).g4hit.size(); ihit++){
        sum_energy_g4hit = sum_energy_g4hit + ((FCS_matchedcell)(*vec)[j]).g4hit[ihit].hit_energy;
      }

      //now I have the hit energy in a given cell, and I store it in a map
      //cout << endl;
      //cout << ((FCS_matchedcell)(*vec)[j]).cell.cell_identifier << endl;
      //cout << sum_energy_hit << endl;
      Eoriginal.insert(std::pair<Long64_t, double>(((FCS_matchedcell)(*vec)[j]).cell.cell_identifier, sum_energy_hit));
      EoriginalReco.insert(std::pair<Long64_t, double>(((FCS_matchedcell)(*vec)[j]).cell.cell_identifier, ((FCS_matchedcell)(*vec)[j]).cell.energy));
      EoriginalG4.insert(std::pair<Long64_t, double>(((FCS_matchedcell)(*vec)[j]).cell.cell_identifier, sum_energy_g4hit));
    } //end of loop over cells of originalTree
    ///////////////////////
     



    //NOW make the average shower shapes by looping over all maps
    for ( std::map<Long64_t, double>::iterator it = Eoriginal.begin(); it!=Eoriginal.end(); it++) {
      
      Long64_t myID = it->first;

      const CaloGeoDetDescrElement* cell;
      Identifier cellid(myID);
      cell=geo->getDDE(cellid);
      
      double dEta = trEta - cell->eta();
      double dPhi = TVector2::Phi_mpi_pi(trPhi - cell->phi());

      dEta_dPhi_FCS_orig[10]->Fill(dEta, dPhi, (it->second));
      dEta_dPhi_FCS_orig[PCAbin]->Fill(dEta, dPhi, (it->second));
        
    }



    for ( std::map<Long64_t, double>::iterator it = EoriginalReco.begin(); it!=EoriginalReco.end(); it++) {
      
      Long64_t myID = it->first;

      const CaloGeoDetDescrElement* cell;
      Identifier cellid(myID);
      cell=geo->getDDE(cellid);
      
      double dEta = trEta - cell->eta();
      double dPhi = TVector2::Phi_mpi_pi(trPhi - cell->phi());

      dEta_dPhi_avg_reco[10]->Fill(dEta, dPhi, (it->second));
      dEta_dPhi_avg_reco[PCAbin]->Fill(dEta, dPhi, (it->second));
        
    }

    for ( std::map<Long64_t, double>::iterator it = EoriginalG4.begin(); it!=EoriginalG4.end(); it++) {
      
      Long64_t myID = it->first;

      const CaloGeoDetDescrElement* cell;
      Identifier cellid(myID);
      cell=geo->getDDE(cellid);
      
      double dEta = trEta - cell->eta();
      double dPhi = TVector2::Phi_mpi_pi(trPhi - cell->phi());

      dEta_dPhi_avg_g4[10]->Fill(dEta, dPhi, (it->second));
      dEta_dPhi_avg_g4[PCAbin]->Fill(dEta, dPhi, (it->second));
        
    }

  } //end of event loop
  



  //  TH2F * cl_ratio = (TH2F*)dEta_dPhi[0]->Clone("cl_ratio_"+sampling);
  //cl_ratio->Divide(dEta_dPhi_N[0]);
  //cl_ratio->GetXaxis()->SetTitle("d#eta");
  //cl_ratio->GetYaxis()->SetTitle("d#phi");
  //cl_ratio->Draw("colz");

  //Now I have to run over events again to get the ratio to the average 
  for (Int_t ientry=0; ientry<nEvt; ientry++){
    
    inputTree->GetEntry(ientry);   
    if(ientry%100 == 0)  
      std::cout << "Processing event # " << ientry << std::endl;
    PCAtree->GetEntry(ientry);

    Eoriginal.clear();
    EoriginalG4.clear();
    EoriginalReco.clear();
    Eclosure.clear();

    double trEta = 0.0;
    double trPhi = 0.0;

    trEta = (*truth)[0].TTC_entrance_eta[0];
    trPhi = (*truth)[0].TTC_entrance_phi[0];

    double sum_energy_hit = 0.0; 

    for (UInt_t j=0; j<(*vec).size(); j++){    
      // sum_energy_hit = 0.0;
      //loop over hits and assigning energy to cells
      for (int ihit=0; ihit<((FCS_matchedcell)(*vec)[j]).hit.size(); ihit++){
	// sum_energy_hit = sum_energy_hit + ((FCS_matchedcell)(*vec)[j]).hit[ihit].hit_energy;
	//now I check what is the hit position
	float x = ((FCS_matchedcell)(*vec)[j]).hit[ihit].hit_x;
	float y = ((FCS_matchedcell)(*vec)[j]).hit[ihit].hit_y;
	float z = ((FCS_matchedcell)(*vec)[j]).hit[ihit].hit_z;
	float t = ((FCS_matchedcell)(*vec)[j]).hit[ihit].hit_time;	
	TLorentzVector *hitVec = new TLorentzVector(x, y, z, t);
	int hitSampling=((FCS_matchedcell)(*vec)[j]).hit[ihit].sampling;
	const CaloGeoDetDescrElement* someCell;
	if(hitSampling<21)someCell=geo->getDDE(hitSampling,hitVec->Eta(),hitVec->Phi());
	else if (hitSampling<24)someCell=geo->getFCalDDE(hitSampling,x,y,z);
	else someCell=0;
	Long64_t someCellID = someCell->identify();
	
	if (Eoriginal.find(someCellID) != Eoriginal.end()){
	  std::map<Long64_t, double>::iterator someit = Eoriginal.find(someCellID);
	  someit->second+=((FCS_matchedcell)(*vec)[j]).hit[ihit].hit_energy;
	} //end cells exist
	
	else {
	  Eoriginal.insert(std::pair<Long64_t, double>(someCellID, ((FCS_matchedcell)(*vec)[j]).hit[ihit].hit_energy));
	}//end of else creating new cell	
      } //end hit loop
      
      //Eoriginal.insert(std::pair<Long64_t, double>(((FCS_matchedcell)(*vec)[j]).cell.cell_identifier, sum_energy_hit));
    }//loop over cells

    
    

    ///////////////////////////////////
    //    Wiggle assignment start    //
    ///////////////////////////////////
   
    
    //loop over cells in sampling
    for (UInt_t j=0; j<(*vec).size(); j++){
      
      Long64_t cell_ID = 0;
      cell_ID = ((FCS_matchedcell)(*vec)[j]).cell.cell_identifier;
      
      //set random numbers
      TRandom3 *random1 = new TRandom3();
      random1->SetSeed(0);
      
      //now I use the geomery lookup tool to get the cell eta/phi
      const CaloGeoDetDescrElement* cell;
      Identifier cellid(cell_ID);
      cell=geo->getDDE(cellid); //This is working also for the FCal
      
      //loop over hits in the cell
      for (int ihit=0; ihit<((FCS_matchedcell)(*vec)[j]).hit.size(); ihit++){
	
	//now I check what is the hit position
	float x = ((FCS_matchedcell)(*vec)[j]).hit[ihit].hit_x;
	float y = ((FCS_matchedcell)(*vec)[j]).hit[ihit].hit_y;
	float z = ((FCS_matchedcell)(*vec)[j]).hit[ihit].hit_z;
	float t = ((FCS_matchedcell)(*vec)[j]).hit[ihit].hit_time;	
	TLorentzVector *hitVec = new TLorentzVector(x, y, z, t);
	int hitSampling=((FCS_matchedcell)(*vec)[j]).hit[ihit].sampling;
	const CaloGeoDetDescrElement* initCell;
	if(hitSampling<21)initCell=geo->getDDE(hitSampling,hitVec->Eta(),hitVec->Phi());
	else if (hitSampling<24)initCell=geo->getFCalDDE(hitSampling,x,y,z);
	else initCell=0;
	
	if (initCell && cell){	  
	  float efficiencyEta = ( (2.0*(hitVec->Eta()-initCell->eta()))/initCell->deta());
	  float efficiencyPhi = ( (2.0*(hitVec->Phi()-initCell->phi()))/initCell->dphi());
	  
	  double searchRand = random1->Rndm(0);
	  //cout << searchRand << endl;
	  int chosenBin = (Int_t) TMath::BinarySearch(50, eff_int, searchRand);
	  //cout << chosenBin << endl; 
	  //cout << "x axis value:  " << eff_phi->GetBinCenter(chosenBin+2) << endl; 
	  //cout << endl;
	  //double wigglePhi = (eff_phi->GetBinCenter(chosenBin+2))/2;
	  double wigglePhi = 0.0;
	  //change the wiggle for the derivative
	  if (sampling=="Sampling_1"){
	    if (efficiencyPhi>0.76){
	      //cout << "plus yoo" << endl;
	      //wigglePhi = (0.24*searchRand);
	      wigglePhi = (eff_phi->GetBinCenter(chosenBin+2))/2;
	    }
	    else if (efficiencyPhi<-0.76){
	      //cout << "minus yoo" << endl;
	      //wigglePhi = -(0.24*searchRand);
	      wigglePhi = (eff_phi->GetBinCenter(chosenBin+2))/2;
	    }/// FAD FAD FAD ///
	    else
	      wigglePhi = 0.0; 
	  }
	  else
	    wigglePhi = (eff_phi->GetBinCenter(chosenBin+2))/2; 
	  
	  const CaloGeoDetDescrElement* foundCell;
	  if(hitSampling<21)foundCell=geo->getDDE(hitSampling,hitVec->Eta(),(hitVec->Phi())-(wigglePhi*initCell->dphi()));
	  else if (hitSampling<24)foundCell=geo->getFCalDDE(hitSampling,x,y,z);
		else foundCell=0;

	  //deposit energy in specific cell
	  Long64_t foundCellID = foundCell->identify();
	  if (Eclosure.find(foundCellID) != Eclosure.end()){
	    std::map<Long64_t, double>::iterator it = Eclosure.find(foundCellID);
	    // found cell iD: foundCell->identify()
	    it->second+=((FCS_matchedcell)(*vec)[j]).hit[ihit].hit_energy;
	  } //end cells exist
	  else {
	    //create the cell --> OK assigned probabilistically although not really in terms of cell IDs
	    Eclosure.insert(std::pair<Long64_t, double>(foundCellID, ((FCS_matchedcell)(*vec)[j]).hit[ihit].hit_energy));
	  }//end of else creating new cell
	} //end if cell ok
      } //end loop over hits
      
    } //end loop on cells


    ///////////////////////////////////
    //     Wiggle assignment end     //
    ///////////////////////////////////

    //now I iterate over my closure file and fill the plots
    for ( std::map<Long64_t, double>::iterator it2 = Eclosure.begin(); it2!=Eclosure.end(); it2++) {
      
      Long64_t myID = it2->first;
      const CaloGeoDetDescrElement* cell;
      Identifier cellid(myID);
      cell=geo->getDDE(cellid);
      
      double dEta = trEta - cell->eta();
      double dPhi = TVector2::Phi_mpi_pi(trPhi - cell->phi());

      Int_t binReco = dEta_dPhi_avg_reco[10]->FindBin(dEta, dPhi);
      double eReco = dEta_dPhi_avg_reco[10]->GetBinContent(binReco);
      Int_t binG4 = dEta_dPhi_avg_g4[10]->FindBin(dEta, dPhi);
      double eG4 = dEta_dPhi_avg_g4[10]->GetBinContent(binG4);

      Int_t binRecoPCA = dEta_dPhi_avg_reco[PCAbin]->FindBin(dEta, dPhi);
      double eRecoPCA = dEta_dPhi_avg_reco[PCAbin]->GetBinContent(binRecoPCA);
      Int_t binG4PCA = dEta_dPhi_avg_g4[PCAbin]->FindBin(dEta, dPhi);
      double eG4PCA = dEta_dPhi_avg_g4[PCAbin]->GetBinContent(binG4PCA);


      dEta_dPhi_FCS_matched_overAvg[10]->Fill(dEta, dPhi, it2->second/eReco);
      dEta_dPhi_FCS_matched_overAvg[PCAbin]->Fill(dEta, dPhi, it2->second/eRecoPCA);

      dEta_dPhi_FCS_matched_overAvg_N[10]->Fill(dEta, dPhi);
      dEta_dPhi_FCS_matched_overAvg_N[PCAbin]->Fill(dEta, dPhi);


    }

    for (std::map<Long64_t, double>::iterator it = Eoriginal.begin(); it!=Eoriginal.end(); it++){

      Long64_t myID = it->first;
      
      const CaloGeoDetDescrElement* cell;
      Identifier cellid(myID);
      cell=geo->getDDE(cellid);
      
      double dEta = trEta - cell->eta();
      double dPhi = TVector2::Phi_mpi_pi(trPhi - cell->phi());

      Int_t binReco = dEta_dPhi_avg_reco[10]->FindBin(dEta, dPhi);
      double eReco = dEta_dPhi_avg_reco[10]->GetBinContent(binReco);
      Int_t binG4 = dEta_dPhi_avg_g4[10]->FindBin(dEta, dPhi);
      double eG4 = dEta_dPhi_avg_g4[10]->GetBinContent(binG4);

      Int_t binRecoPCA = dEta_dPhi_avg_reco[PCAbin]->FindBin(dEta, dPhi);
      double eRecoPCA = dEta_dPhi_avg_reco[PCAbin]->GetBinContent(binRecoPCA);
      Int_t binG4PCA = dEta_dPhi_avg_g4[PCAbin]->FindBin(dEta, dPhi);
      double eG4PCA = dEta_dPhi_avg_g4[PCAbin]->GetBinContent(binG4PCA);

      dEta_dPhi_FCS_orig_overAvg[10]->Fill(dEta, dPhi, it->second/eReco);
      dEta_dPhi_FCS_orig_overAvg[PCAbin]->Fill(dEta, dPhi, it->second/eRecoPCA);
 
      dEta_dPhi_FCS_orig_overAvg_N[10]->Fill(dEta, dPhi);
      dEta_dPhi_FCS_orig_overAvg_N[PCAbin]->Fill(dEta, dPhi);


    }
    

  }//end new event loop

  
  
  //now print it all
  TCanvas *c1 = new TCanvas();
  c1->SetLogz();
  c1->Divide(1,2);
  c1->cd(1);
  dEta_dPhi[0]->Draw("colz");
  c1->cd(2);
  dEta_dPhi_N[0]->Draw("colz");  

  c1->Print("wiggle_plots_ClosureTest"+sampling+".pdf(");


  TCanvas *c4 = new TCanvas();
  c4->Divide(1,2);
  c4->cd(1);
  dEta_dPhi_res[0]->Draw("colz");
  c4->cd(2);
  dEta_dPhi_N_res[0]->Draw("colz");  

  c4->Print("wiggle_plots_ClosureTest"+sampling+".pdf");


  TCanvas *c2[11];
  TCanvas *c3[11];

  TCanvas *cc2[11];
  TCanvas *cc3[11];

  TCanvas *ccc2[11];
  TCanvas *ccc3[11];

  TCanvas *ccc4[11];
  TCanvas *ccc5[11];
  TCanvas *ccc6[11];

  for (int i=0; i<11; i++){
    //    c2[i]= new TCanvas();
    c2[i] = new TCanvas("c2"+i,"c2"+i,2);

    c2[i]->Divide(2,2);
    c2[i]->cd(1);
    dEta_dPhi_avg_reco[i]->Draw("colz");
    c2[i]->cd(2);
    dEta_dPhi_avg_g4[i]->Draw("colz");
    c2[i]->cd(3);
    dEta_dPhi_FCS_orig[i]->Draw("colz");
    c2[i]->cd(4);
    dEta_dPhi_FCS_matched[i]->Draw("colz");

    c2[i]->Print("wiggle_plots_ClosureTest"+sampling+".pdf");


    //c3[i]= new TCanvas();
    c3[i] = new TCanvas("c3"+i,"c3"+i,2);

    c3[i]->Divide(2,2);
    c3[i]->cd(1);
    dEta_dPhi_avg_reco[i]->GetXaxis()->SetRangeUser(-0.1,0.1);
    dEta_dPhi_avg_reco[i]->GetYaxis()->SetRangeUser(-0.1,0.1);
    dEta_dPhi_avg_reco[i]->Draw("colz");
    c3[i]->cd(2);
    dEta_dPhi_avg_g4[i]->GetXaxis()->SetRangeUser(-0.1,0.1);
    dEta_dPhi_avg_g4[i]->GetYaxis()->SetRangeUser(-0.1,0.1);
    dEta_dPhi_avg_g4[i]->Draw("colz");
    c3[i]->cd(3);
    dEta_dPhi_FCS_orig[i]->GetXaxis()->SetRangeUser(-0.1,0.1);
    dEta_dPhi_FCS_orig[i]->GetYaxis()->SetRangeUser(-0.1,0.1);
    dEta_dPhi_FCS_orig[i]->Draw("colz");
    c3[i]->cd(4);
    dEta_dPhi_FCS_matched[i]->GetXaxis()->SetRangeUser(-0.1,0.1);
    dEta_dPhi_FCS_matched[i]->GetYaxis()->SetRangeUser(-0.1,0.1);
    dEta_dPhi_FCS_matched[i]->Draw("colz");

    c3[i]->Print("wiggle_plots_ClosureTest"+sampling+".pdf");

    //cc2[i] = new TCanvas();
    cc2[i] = new TCanvas("cc22"+i,"cc22"+i,2);


    //cc2[i]->Divide(2,1);
    //cc2[i]->cd(1);
    
    //%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    // to make bins zero if less than 10 entries
    //%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    /*
    for (int ki=1; ki<nbins_eta+1; ki++){
      for (int j=1; j<nbins_phi+1; j++){

	if (dEta_dPhi_FCS_matched_overAvg_N[10]->GetBinContent(dEta_dPhi_FCS_matched_overAvg_N[10]->GetBin(ki,j)) < 10){	  
	  dEta_dPhi_FCS_matched_overAvg[10]->SetBinContent(dEta_dPhi_FCS_matched_overAvg[10]->GetBin(ki,j), 0.0);
	}

	if (dEta_dPhi_FCS_orig_overAvg_N[10]->GetBinContent(dEta_dPhi_FCS_orig_overAvg_N[10]->GetBin(ki,j)) < 10){	  
	  dEta_dPhi_FCS_orig_overAvg[10]->SetBinContent(dEta_dPhi_FCS_orig_overAvg[10]->GetBin(ki,j), 0.0);
	}



      }
      }*/
    //%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    //cc2[i]->SetLogz();
    //dEta_dPhi_FCS_orig_overAvg[i]->GetZaxis()->SetRangeUser(0.1,10);
    //dEta_dPhi_FCS_orig_overAvg[i]->Rebin2D(nrebin);
    dEta_dPhi_FCS_orig_overAvg[i]->Draw("colz");



    //ccc2[i] = new TCanvas();
    ccc2[i] = new TCanvas("cc2"+i,"cc2"+i,2);

    //ccc2[i]->SetLogz();
    //dEta_dPhi_FCS_matched_overAvg[i]->GetZaxis()->SetRangeUser(0.1,10);
    //dEta_dPhi_FCS_matched_overAvg[i]->Rebin2D(nrebin);
    dEta_dPhi_FCS_matched_overAvg[i]->Draw("colz");


    cc2[i]->Print("wiggle_plots_ClosureTest"+sampling+".pdf");
    ccc2[i]->Print("wiggle_plots_ClosureTest"+sampling+".pdf");

    //   ccc3[i] = new TCanvas();
    ccc3[i] = new TCanvas("cc3"+i,"cc3"+i,2);

    subtraction[i] = (TProfile2D*)dEta_dPhi_FCS_orig_overAvg[i]->Clone("subtraction_"+sampling);
    subtraction[i]->Add(dEta_dPhi_FCS_matched_overAvg[i],-1);
    subtraction[i]->Draw("colz");

    ccc3[i]->Print("wiggle_plots_ClosureTest"+sampling+".pdf");


    //    ccc4[i] = new TCanvas();
    ccc4[i] = new TCanvas("cc4"+i,"cc4"+i,2);

    //ccc4[i]->SetLogz();
    dEta_dPhi_FCS_orig_overAvg[i]->GetXaxis()->SetRangeUser(-0.1,0.1);
    dEta_dPhi_FCS_orig_overAvg[i]->GetYaxis()->SetRangeUser(-0.1,0.1);
    dEta_dPhi_FCS_orig_overAvg[i]->GetZaxis()->SetRangeUser(0.5,2.0);
    dEta_dPhi_FCS_orig_overAvg[i]->Draw("colz");
    ccc4[i]->Print("wiggle_plots_ClosureTest"+sampling+".pdf");

    //ccc5[i] = new TCanvas();
    ccc5[i] = new TCanvas("cc5"+i,"cc5"+i,2);

    //ccc5[i]->SetLogz();
    dEta_dPhi_FCS_matched_overAvg[i]->GetXaxis()->SetRangeUser(-0.1,0.1);
    dEta_dPhi_FCS_matched_overAvg[i]->GetYaxis()->SetRangeUser(-0.1,0.1);
    dEta_dPhi_FCS_matched_overAvg[i]->GetZaxis()->SetRangeUser(0.5,2.0);
    dEta_dPhi_FCS_matched_overAvg[i]->Draw("colz");
    ccc5[i]->Print("wiggle_plots_ClosureTest"+sampling+".pdf");

    ccc6[i] = new TCanvas("cc6"+i,"cc6"+i,2);
    subtraction[i]->GetXaxis()->SetRangeUser(-0.1,0.1);
    subtraction[i]->GetYaxis()->SetRangeUser(-0.1,0.1);
    subtraction[i]->Draw("colz");
    if (i==10){
      ccc6[i]->Print("wiggle_plots_ClosureTest"+sampling+".pdf)");
    }
    else {
      ccc6[i]->Print("wiggle_plots_ClosureTest"+sampling+".pdf");
    }

  } // i loop

} //end of main loop
