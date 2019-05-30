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
#include "../ISF_FastCaloSimParametrization/FCAL_ChannelMap.h"
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

void TestFCalIdentifiers(TString sampling="Sampling_0"){

  CaloGeometryFromFile* geo=new CaloGeometryFromFile();
  geo->LoadGeometryFromFile("/afs/cern.ch/atlas/groups/Simulation/FastCaloSim/ATLAS-GEO-20-00-01.root","ATLAS-GEO-20-00-01");
  geo->LoadFCalGeometryFromFiles("FCal1-electrodes.sorted.HV.09Nov2007.dat","FCal2-electrodes.sorted.HV.April2011.dat","FCal3-electrodes.sorted.HV.09Nov2007.dat");
  
  
  FCAL_ChannelMap* channelMap =geo->GetFCAL_ChannelMap();
  
  
  
  
  
  
  
  
  
  
  
  
  //TFile *inputFile = TFile::Open("root://eosatlas//eos/atlas/user/z/zhubacek/FastCaloSim/LArShift020715/ISF_HitAnalysis6_evgen_calo__211_E50000_50000_eta20_25_Evts0-5500_vz_0_origin_calo.merged.pool.root");
  
  //  TFile *inputFile = TFile::Open("root://eosatlas//eos/atlas/user/z/zhubacek/FastCaloSim/NTUP_110216/ISF_HitAnalysis_Zach1_merged.root");
	
	TFile *inputFile = TFile::Open("/eos/atlas/user/s/schaarsc/FCS/user.fladias.428137.FastCalo_pid11_E65536_etam35_35_zv_m100.e4001_s2864_r7736.w0_162706_matched_output.root/user.fladias.8834798._000001.matched_output.root");
	//TFile *inputFile = TFile::Open("/eos/atlas/user/s/schaarsc/FCS/user.fladias.428137.FastCalo_pid11_E65536_etam35_35_zv_m100.e4001_s2865_r7736.w0_162706_matched_output.root");

  TTree *inputTree = ( TTree* ) inputFile->Get( "FCS_ParametrizationInput" );

  FCS_matchedcellvector *vec=0; //this need to be set to 0!
  inputTree->SetBranchAddress(sampling,&vec);


  std::cout << "Sampling is " << sampling << std::endl;

  Int_t nEvt = inputTree->GetEntries();
  std::cout << "nEvt " << nEvt << std::endl;
  
 // nEvt = 1000;
  
  stringstream ssIdentifier;
  
  for (Int_t ientry=0; ientry<nEvt; ientry++){
    
    inputTree->GetEntry(ientry);   
    if(ientry%100 == 0)  
      std::cout << "Processing event # " << ientry << std::endl;

    double weight = 1.0;
    //cout << (*vec).size() << endl;
    //loop over cells in sampling
    for (UInt_t j=0; j<(*vec).size(); j++){
      
      Float_t cell_E = 0.0;
      cell_E = ((FCS_matchedcell)(*vec)[j]).cell.energy;
      Long64_t cell_ID = 0;
      cell_ID =  ((FCS_matchedcell)(*vec)[j]).cell.cell_identifier;
			
			cout << cell_ID << hex << " 0x"<< cell_ID << endl;
     // cout << cell_E << endl;
      
      //now I use the geomery lookup tool to get the cell eta/phi
     // const CaloDetDescrElement* cell;
      Identifier cellid(cell_ID);
      //cell=geo->getDDE(cellid); //This is working also for the FCal
      
      //loop over hits in the cell
      for (unsigned int ihit=0; ihit<((FCS_matchedcell)(*vec)[j]).hit.size(); ihit++){
	
				//now I check what is the hit position
				float x = ((FCS_matchedcell)(*vec)[j]).hit[ihit].hit_x;
				float y = ((FCS_matchedcell)(*vec)[j]).hit[ihit].hit_y;
				float z = ((FCS_matchedcell)(*vec)[j]).hit[ihit].hit_z;
				float t = ((FCS_matchedcell)(*vec)[j]).hit[ihit].hit_time;	
				TLorentzVector *hitVec = new TLorentzVector(x, y, z, t);
				const CaloDetDescrElement* foundCell;
				int hitSampling =((FCS_matchedcell)(*vec)[j]).hit[ihit].sampling; 
				
				//cout << hitSampling << " " << x << " " << y << endl;
				
				if(hitSampling<21)foundCell=geo->getDDE(hitSampling,hitVec->Eta(),hitVec->Phi());
				else if(hitSampling<24)foundCell=geo->getFCalDDE(hitSampling,x,y,z);
				else {
					cout << endl << "Warning: Found hit with sampling > 23 !!!!!!!!!!!!!!" << endl << endl;
					foundCell =0;
				}
				
				
				int ieta,iphi;
				
				//cout << "Hit position: " << "x: " << x << " y: " << y << " eta: " << hitVec->Eta() << " phi: " << hitVec->Phi() <<  endl;
				
				
				
				if (foundCell){// && cell){
					channelMap->getTileID(hitSampling - 20,x,y,ieta,iphi);
					ssIdentifier.str("");
					
					int zSide = 2;
					ssIdentifier << 4          // LArCalorimeter
             << 3          // LArFCAL
             << zSide      // EndCap
             << hitSampling   // FCal Module # 
             << (ieta << 16)+iphi;  
					
					//cout << foundCell->x() << " " << foundCell->y()  << endl;
					//cout << abs(foundCell->x() - x) << " " << abs(foundCell->y() - y) << endl;
				  if( z > 0) cout << "Side A " << hex << foundCell->identify()  <<" " << cell_ID <<" " << ((FCS_matchedcell)(*vec)[j]).hit[ihit].cell_identifier << dec << " " << cell_ID <<" " << ((FCS_matchedcell)(*vec)[j]).hit[ihit].cell_identifier <<" " <<  ieta << " " << iphi << " " << (ieta << 16)+iphi << endl;
				  else cout << "Side C " << hex << foundCell->identify()  <<" " << cell_ID << " " << ((FCS_matchedcell)(*vec)[j]).hit[ihit].cell_identifier << dec<< " " << cell_ID <<" " << ((FCS_matchedcell)(*vec)[j]).hit[ihit].cell_identifier << " " <<  ieta << " " << iphi << " " << (ieta << 16)+iphi << endl;
				 // cout << cell_ID << " " << ((FCS_matchedcell)(*vec)[j]).hit[ihit].identifier << " " << ((FCS_matchedcell)(*vec)[j]).hit[ihit].cell_identifier << " " << ((FCS_matchedcell)(*vec)[j]).g4hit[ihit].identifier << " " << ((FCS_matchedcell)(*vec)[j]).g4hit[ihit].cell_identifier << endl;
				 
				  
				} //end if cell ok
				else cout << "Cell not found!" << endl;
      } //end loop over hits
    } //end loop on cells   
  } //end loop on events
  
  



}
