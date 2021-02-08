/*
  Copyright (C) 2002-2017 CERN for the benefit of the ATLAS collaboration
*/

#define PlotHits_cxx

#include "fBinning.C"
#include "TLorentzVector.h"
#include <iomanip>
#include "TH1F.h"
#include "TFile.h"
#include "TPad.h"
#include <iostream>
#include <fstream>
#include <iterator>
#include <vector>
#include <stdio.h>
#include <string.h>
#include <stddef.h>
#include <string>
#include "TGraph.h"
#include "TGraphPolar.h"
#include "TStyle.h"
#include "TMatrixD.h"
#include "TLine.h"
#include "loader.C"
#include <iostream>
#include <fstream>
#include <iterator>
#include <vector>
#include <stdio.h>
#include <string.h>
#include <stddef.h>
#include <string>
#include "TMath.h"
#include "TH2F.h"
#include "TH1F.h"
#include "TFile.h"
#include "TTree.h"
#include "TLeaf.h"
#include "TVector3.h"
#include "TChain.h"
#include <iostream>
#include "TDirectory.h"
#include "TROOT.h"
#include "TCanvas.h"
#include <string>
#include <sstream>
#include "TLegend.h"
#include "TPie.h"
#include <stdio.h>
#include <stdlib.h>


using namespace std;

vector<float> fVecBin(TH1F myh, int mybin, bool doR);


int main(int argc, char **argv){

  //int ccTTC=0;

  //ofstream of;
  //of.open("CellIdsWithHighdr.txt");

  TH1::SetDefaultSumw2(kTRUE);

  bool doexp=false;


  bool doDebug = true ; 

  float tolerance = 0.00001;
  float nrelvar1 = 0.5;
  float nrelvar2 = 0.7;

  string outdir = "/afs/cern.ch/work/c/conti/private/AF2/OUTPUT" ;
  string doPart = "pi";
  float selected_layer = 1;
  ostringstream os;
  os << selected_layer ;

  for(int i2=0;i2<argc;i2++){
    if (!strcmp(argv[i2], "--part"))       doPart = argv[i2+1];
    if (!strcmp(argv[i2], "--outdir"))     outdir = argv[i2+1];
    if (!strcmp(argv[i2], "--layer") )     selected_layer = atof(argv[i2+1]);
  }

  std::cout << "Settings :: particle : " << doPart << "  layer : " << selected_layer << std::endl;
  TH1::SetDefaultSumw2(kTRUE);
  gROOT->ProcessLine("#include <vector>");
  gStyle->SetOptStat(0) ; 

  // BINNING CHRISTOPHER ###############################################################
  
  const Int_t nxbins  = 29 ; 
  const Int_t nybins  = 29 ;
  Float_t xbins[nxbins]   = {-0.4,-0.1,-0.05,-0.04,-0.03,-0.02,-0.016,-0.014,-0.012,-0.01,-0.008,-0.006,-0.004,-0.002,0,0.002,0.004,0.006,0.008,0.01,0.012,0.014,0.016,0.02,0.03,0.04,0.05,0.1,0.4} ;
  Float_t ybins[nybins]   = {-0.4,-0.1,-0.05,-0.04,-0.03,-0.02,-0.016,-0.014,-0.012,-0.01,-0.008,-0.006,-0.004,-0.002,0,0.002,0.004,0.006,0.008,0.01,0.012,0.014,0.016,0.02,0.03,0.04,0.05,0.1,0.4} ;

  // POSITIVE BINNING IN DR (From Christopher's numbers) 
  const Int_t nybinsPos = 15  ;
  Float_t ybinsPos[nybinsPos] = {0,0.002,0.004,0.006,0.008,0.01,0.012,0.014,0.016,0.02,0.03,0.04,0.05,0.1,0.4} ;

  // REGULAR BINNIN IN ALPHA
  const Int_t nxbinsReg =  9 ;
  float rotZach = TMath::Pi()/8. ; 
  Float_t xbinsReg[nxbinsReg] = {0+rotZach,0.25*TMath::Pi()+rotZach,0.5*TMath::Pi()+rotZach,0.75*TMath::Pi()+rotZach,TMath::Pi()+rotZach,1.25*TMath::Pi()+rotZach,1.5*TMath::Pi()+rotZach,1.75*TMath::Pi()+rotZach,2*TMath::Pi()+rotZach};

  // INPUT FILE #######################################################################

  string myfile = ""; 
  if(doPart=="e")
    myfile = "INPUTS/ISF_HitAnalysisESD_evgen_calo__11__E50000_50000__eta20_25_Evts0-5500_zvertex_0.pool.root" ;
  if(doPart=="pi")
    //myfile = "root://eosatlas.cern.ch//eos/atlas/user/z/zhubacek/FastCaloSim/NTUP_061114/OUTPUT_Merge_evgen_calo__211_E50000_50000_eta20_25_Evts0-5500_vz_0_origin_calo.pool.root/ISF_HitAnalysis_evgen_calo__211_E50000_50000_eta20_25_Evts0-5500_vz_0_origin_calo.pool.root" ;
    myfile = "INPUTS/ISF_HitAnalysis_evgen_calo__211_E50000_50000_eta20_25_Evts0-5500_vz_0_origin_calo.pool.root" ; 
    //myfile = "INPUTS/ISF_HitAnalysisESD_evgen_calo__211__E50000_50000__eta20_25_Evts0-5500_zvertex_0.pool.root" ; 

  // PIONS 
  // 10 GeV 
  //eos/atlas/user/z/zhubacek/FastCaloSim/NTUP_061114/OUTPUT_Merge_evgen_calo__211_E10000_10000_eta200_205_Evts0-5500_vz_0_origin_calo.pool.root/ISF_HitAnalysis_evgen_calo__211_E10000_10000_eta200_205_Evts0-5500_vz_0_origin_calo.pool.merged.root
  //eos/atlas/user/z/zhubacek/FastCaloSim/NTUP_061114/OUTPUT_Merge_evgen_calo__211_E10000_10000_eta20_25_Evts0-5500_vz_-100_origin_calo.pool.root/ISF_HitAnalysis_evgen_calo__211_E10000_10000_eta20_25_Evts0-5500_vz_-100_origin_calo.pool.merged.root
  //eos/atlas/user/z/zhubacek/FastCaloSim/NTUP_061114/OUTPUT_Merge_evgen_calo__211_E10000_10000_eta20_25_Evts0-5500_vz_0_origin_calo.pool.root/ISF_HitAnalysis_evgen_calo__211_E10000_10000_eta20_25_Evts0-5500_vz_0_origin_calo.pool.merged.root
  //eos/atlas/user/z/zhubacek/FastCaloSim/NTUP_061114/OUTPUT_Merge_evgen_calo__211_E10000_10000_eta20_25_Evts0-5500_vz_100_origin_calo.pool.root/ISF_HitAnalysis_evgen_calo__211_E10000_10000_eta20_25_Evts0-5500_vz_100_origin_calo.pool.merged.root
  //eos/atlas/user/z/zhubacek/FastCaloSim/NTUP_061114/OUTPUT_Merge_evgen_calo__211_E10000_10000_eta400_405_Evts0-5500_vz_0_origin_calo.pool.root/ISF_HitAnalysis_evgen_calo__211_E10000_10000_eta400_405_Evts0-5500_vz_0_origin_calo.pool.merged.root
  // 50 GeV
  //eos/atlas/user/z/zhubacek/FastCaloSim/NTUP_061114/OUTPUT_Merge_evgen_calo__211_E50000_50000_eta200_205_Evts0-5500_vz_0_origin_calo.pool.root/ISF_HitAnalysis_evgen_calo__211_E50000_50000_eta200_205_Evts0-5500_vz_0_origin_calo.pool.merged.root
  //eos/atlas/user/z/zhubacek/FastCaloSim/NTUP_061114/OUTPUT_Merge_evgen_calo__211_E50000_50000_eta20_25_Evts0-5500_vz_0_origin_calo.pool.root/ISF_HitAnalysis_evgen_calo__211_E50000_50000_eta20_25_Evts0-5500_vz_0_origin_calo.pool.merged.root
  //eos/atlas/user/z/zhubacek/FastCaloSim/NTUP_061114/OUTPUT_Merge_evgen_calo__211_E50000_50000_eta20_25_Evts0-5500_vz_100_origin_calo.pool.root/ISF_HitAnalysis_evgen_calo__211_E50000_50000_eta20_25_Evts0-5500_vz_100_origin_calo.pool.merged.root
  //eos/atlas/user/z/zhubacek/FastCaloSim/NTUP_061114/OUTPUT_Merge_evgen_calo__211_E50000_50000_eta400_405_Evts0-5500_vz_0_origin_calo.pool.root/ISF_HitAnalysis_evgen_calo__211_E50000_50000_eta400_405_Evts0-5500_vz_0_origin_calo.pool.merged.root

  TFile* F       = TFile::Open(myfile.c_str()); 
  TTree *T       = (TTree*)F->Get("ISF_HitAnalysis/CaloHitAna");

  // OUTPUT FILE #######################################################################
  
  //TH2F* hdhdfmm = new TH2F("hdhdfmm","hdhdfmm",30,0.5,0.7,30,-0.2,0.2) ;
  //hdhdfmm->GetXaxis()->SetTitle("#delta#eta [mm]");
  //hdhdfmm->GetYaxis()->SetTitle("#delta#phi [mm]");

  //TH2F* hdhdfTTC = new TH2F("hdhdfTTC","hdhdfTTC",600,-3,3,600,-3,3) ;
  //hdhdfTTC->GetXaxis()->SetTitle("#delta#eta");
  //hdhdfTTC->GetYaxis()->SetTitle("#delta#phi");


  // CHRISTOPHER BINNING ##############################################################################

  std::cout << "########################################################## " << std::endl;
  std::cout << "1. CHRISTOPHER BINNING ################################### " << std::endl; 

  TH1F halpha("halpha","halpha",7500000,0+rotZach,2*TMath::Pi()+rotZach) ;
  halpha.GetXaxis()->SetTitle("#alpha");

  TH1F hdr("hdr","hdr",7500000,0,0.4) ;
  hdr.GetXaxis()->SetTitle("dr");
  
  TH1F halphaE("halphaE","halphaE",nxbinsReg-1,xbinsReg) ;
  halphaE.GetXaxis()->SetTitle("#alpha");

  TH1F hdrE("hdrE","hdrE",nybinsPos-1,ybinsPos) ;
  hdrE.GetXaxis()->SetTitle("#delta r");

  TMatrixD mEnergy(nxbinsReg-1,nybinsPos-1);
  TMatrixD mxEnergy(nxbinsReg-1,nybinsPos-1);
  TMatrixD myEnergy(nxbinsReg-1,nybinsPos-1);

  TH2F halphadrE = fBinning(myfile,nxbins,xbins,nybins,ybins,
			    nxbinsReg, xbinsReg,nybinsPos, ybinsPos, 
			    "Christopher",selected_layer,doPart,
			    halpha,hdr,halphaE,hdrE,
			    mEnergy,mxEnergy,myEnergy);

  // Rebin in Alpha ####################################################################################
  
  std::cout << "########################################################## " <<std::endl;
  std::cout << "2. ALPHA REBINNING ######################################## " << std::endl; 

  float nAlphaBins          = 8 ; // PARAMETER TO TUNE 

  const int nAlphaBinsVal=nAlphaBins+1 ; Float_t vAlphaBins[nAlphaBinsVal] ; bool doR = false;

  std::vector<float> vAlphaBinsSTD = fVecBin(halpha,nAlphaBins,doR);

  std::cout << "SIZE OF ALPHA STD : " << vAlphaBinsSTD.size() << std::endl;

  for(Int_t i=0;i<nAlphaBinsVal;i++){
    vAlphaBins[i] = vAlphaBinsSTD.at(i) ;
    std::cout << i << "  " << vAlphaBins[i] << std::endl;
  }
  
  // Rebin in R ########################################################################################

  std::cout << "########################################################## " <<std::endl;
  std::cout << "3. R REBINNING ########################################### " << std::endl;
    
  Int_t rmax      =50 ; // PARAMETER TO TUNE 

  Int_t incr      = 0 ;
  bool isEmptyBin = true;
  float nRBins    = 0 ; 
  
  TH1F halphaE2("halphaE2","halphaE2",nAlphaBins,vAlphaBins) ;
  halphaE2.GetXaxis()->SetTitle("#alpha");

  while(isEmptyBin){

    nRBins = rmax-incr ; 
    doR = true ; 
    std::vector<float> vRBinsSTD = fVecBin(hdr,nRBins, doR);
    std::cout << "NUMBER OF R BINS TESTED : " << nRBins << std::endl;
    
    //for(Int_t i=0;i<vRBinsSTD.size();i++){
    //  std::cout << "CEHCK : " << i << "  " << vRBinsSTD.at(i) << std::endl;
    //}

    const int nRBinsVal  = vRBinsSTD.size() ; 
    nRBins = nRBinsVal - 1 ; 
    Float_t vRBins[nRBinsVal];
    for(Int_t i=0;i<nRBinsVal;i++){
      vRBins[i] = vRBinsSTD.at(i) ;
    }
    
    TH1F halpha2("halpha2","halpha2",nAlphaBins,vAlphaBins) ;
    halpha2.GetXaxis()->SetTitle("#alpha");

    TH1F hdr2("hdr2","hdr2",nRBins,vRBins) ;
    hdr2.GetXaxis()->SetTitle("#delta r");

    TH1F hdrE2("hdrE2","hdrE2",nRBins,vRBins) ;
    hdrE2.GetXaxis()->SetTitle("#delta r");

    TMatrixD mEnergy2(nAlphaBins,nRBins);
    TMatrixD mxEnergy2(nAlphaBins,nRBins);
    TMatrixD myEnergy2(nAlphaBins,nRBins);

    TH2F halphadrE2 = fBinning(myfile,nxbins,xbins,nybins,ybins,
				   nAlphaBinsVal,vAlphaBins,nRBinsVal,vRBins,
				   "AlphaBinned",selected_layer,doPart,
				   halpha2,hdr2,halphaE2,hdrE2,
				   mEnergy2,mxEnergy2,myEnergy2);

    // TEST EMPTY BINS #####################################

    std::cout << "START TESTING EMPTY BINS" << std::endl;
    Int_t nprob = 0 ;
    for(Int_t i=1;i<halphadrE2.GetNbinsX()+1;i++){
      std::cout << "PHI BIN : " << i << "   ############## " << std::endl;
      for(Int_t j=1;j<halphadrE2.GetNbinsY()+1;j++){
        if(halphadrE2.GetBinContent(i,j)<tolerance){
	  std::cout << "EMPTY BIN (" << i << "," << j << "), checking new bin number" << std::endl;
          isEmptyBin = true;
          nprob++;
        }
	// Check that the energy variation is lower than 0.30 in the radial direction 
        if(j>1){
	  float relvar = fabs((TMath::Log(halphadrE2.GetBinContent(i,j))-TMath::Log(halphadrE2.GetBinContent(i,j-1)))/TMath::Log(halphadrE2.GetBinContent(i,j-1)));
	  std::cout << j << " :  " << TMath::Log(halphadrE2.GetBinContent(i,j)) << "  " << TMath::Log(halphadrE2.GetBinContent(i,j-1)) << "  " << relvar << std::endl; 

	  if((relvar>nrelvar1 && j<0.5*halphadrE2.GetNbinsY()) || (relvar>nrelvar2 && j>=0.5*halphadrE2.GetNbinsY()) ){
	    std::cout << "TOO LARGE E VARIATION BIN (" << i << "," << j << ") : " << relvar <<", checking new bin number" << std::endl;
	    isEmptyBin = true;
	    nprob++;
	  }
	}
	if(nprob>0) break;
      }
      if(nprob>0) break;
    }
    std::cout << "END OF TESTING EMPTY BINS" << std::endl;
    if(nprob==0) isEmptyBin = false;

    incr++;
    halphadrE2.Reset();

  } // while(isEmptybins)
  
  std::cout << "THE FINAL NUMBER OF BINS IN R IS : " << nRBins << std::endl;

  //#############################################################

  std::cout << "########################################################## " <<std::endl;  
  std::cout << "4. FINAL R BINNING ############################ "  << std::endl;

  // Need to recreate the vector vRBins outside of the loop with the latest number of dr bins 
  doR= true ;
  std::vector<float> vRBinsSTD = fVecBin(hdr,nRBins, doR);
  for(Int_t i=0;i<vRBinsSTD.size();i++){
  }

  const int nRBinsVal  = nRBins + 1 ;
  Float_t vRBins[nRBinsVal];

  //const int nRBinsSym  = nRBins*2 + 1 ;
  //Float_t vRBinsSym[nRBinsSym];

  // Symmetrize the R vector
  for(Int_t i=0;i<nRBinsVal;i++){
    vRBins[i] = vRBinsSTD.at(i) ;
    //vRBinsSym[i]= - vRBinsSTD.at(nRBinsVal-i-1);
    //if(i<nRBinsVal-1) vRBinsSym[i+nRBinsVal] = vRBinsSTD.at(i+1) ;
  }

  std::cout << "R BINNNING " << std::endl;
  for(Int_t i=0;i<nRBinsVal;i++){
    std::cout << i << "  " << vRBins[i] << std::endl;
  }
   
  //std::cout << "R SYMMETRIC BINNNING " << std::endl;
  //for(Int_t i=0;i<nRBinsSym;i++){
  //  std::cout << i << "  " << vRBinsSym[i] << std::endl;
  //}
  
  // CHECK LAST BIN ############################################

  std::cout << "########################################################## " <<std::endl;
  std::cout << "5. CHECK LAST R BIN ###################################### " << std::endl;

  float checklastbin = vRBins[nRBinsVal-1]-vRBins[nRBinsVal-2];
  Int_t nRsubBins     = 1 ;   

  // max value of last bin to be 0.1 
  if(checklastbin/0.1 > 1){
    nRsubBins = int(checklastbin/0.1)+1;
  } 

  Int_t incrlast      = 0 ; 
  bool isEmptyBinlast = true ;
  Int_t nRBinslast    = 1 ; 

  while(isEmptyBinlast){

    nRBinslast             = nRsubBins-incrlast ;
    const int nRBins2last  = nRBinslast + 1 ;
    
    // vector containing the values delimiting the last sub-bins
    Float_t vRBinslast[nRBins2last];
    vRBinslast[0]               = vRBins[nRBinsVal-2]; // start of last bin => start of last sub-bins
    vRBinslast[nRBins2last-1]   = vRBins[nRBinsVal-1]; // end of last bin   => end of last sub-bins

    std::cout << "0 " << vRBinslast[0] << std::endl;
    for(Int_t i=1;i<nRBins2last-1;i++){

      if(!doexp)
	vRBinslast[i] = vRBinslast[0] + i*((vRBinslast[nRBins2last-1]-vRBinslast[0])/nRBinslast); 
      if(doexp)
	vRBinslast[i] = vRBinslast[0] + (i*((vRBinslast[nRBins2last-1]-vRBinslast[0])/nRBinslast))/(nRBins2last-i); 

      std::cout << i << "  " << vRBinslast[i] << std::endl;
    }
    std::cout << nRBins2last-1 << "  " << vRBinslast[nRBins2last-1] << std::endl;

    // vector containing both the vRBins and the vRBinslast vector    
    Float_t vRBinsAll[nRBins2last+nRBinsVal-2];
    for(Int_t i=0;i<nRBins2last+nRBinsVal-2;i++){
      if(i<nRBinsVal-2)   vRBinsAll[i] = vRBins[i];
      if(i>=nRBinsVal-2)  vRBinsAll[i] = vRBinslast[i-nRBinsVal+2];
    }

    TH1F hdr3("hdr3","hdr3",nRBins+nRBinslast-1,vRBinsAll) ;
    hdr3.GetXaxis()->SetTitle("#delta r");

    TH1F halpha3("halpha3","halpha3",nAlphaBins,vAlphaBins) ;
    halpha3.GetXaxis()->SetTitle("#alpha");

    TH1F hdrE3("hdrE3","hdrE3",nRBins+nRBinslast-1,vRBinsAll) ;
    hdrE3.GetXaxis()->SetTitle("#delta r");

    TH1F halphaE3("halphaE3","halphaE3",nAlphaBins,vAlphaBins) ;
    halphaE3.GetXaxis()->SetTitle("#alpha");

    TMatrixD mEnergy3(nAlphaBins,nRBins+nRBinslast-1);
    TMatrixD mxEnergy3(nAlphaBins,nRBins+nRBinslast-1);
    TMatrixD myEnergy3(nAlphaBins,nRBins+nRBinslast-1);

    TH2F halphadrE3 = fBinning(myfile,nxbins,xbins,nybins,ybins,
			       nAlphaBins+1,vAlphaBins,nRBins+nRBinslast,vRBinsAll, 
			       "Rlast",selected_layer,doPart,
			       halpha3,hdr3,halphaE3,hdrE3,
			       mEnergy3,mxEnergy3,myEnergy3);


    std::cout << "START TESTING EMPTY BINS" << std::endl;
    Int_t nproblast = 0 ;
    for(Int_t i=1;i<halphadrE3.GetNbinsX()+1;i++){
      for(Int_t j=1;j<halphadrE3.GetNbinsY()+1;j++){
	if(halphadrE3.GetBinContent(i,j)<tolerance){
	  std::cout << "PROBLEM WITH BIN (" << i << "," << j << ") : " << halphadrE3.GetBinContent(i,j) <<  std::endl;
	  isEmptyBinlast = true;
	  nproblast++;
	}
        if(j>1){
          float relvar = fabs((TMath::Log(halphadrE3.GetBinContent(i,j))-TMath::Log(halphadrE3.GetBinContent(i,j-1)))/TMath::Log(halphadrE3.GetBinContent(i,j-1)));
          if((relvar>nrelvar1 && j<0.5*halphadrE3.GetNbinsY()) || (relvar>nrelvar2 && j>=0.5*halphadrE3.GetNbinsY()) ){
	    std::cout << "TOO LARGE E VARIATION BIN (" << i << "," << j << ") : " << relvar <<", checking new bin number" << std::endl;
            isEmptyBinlast = true;
            nproblast++;
          }
        }
	if(nproblast>0) break;
      }
      if(nproblast>0) break;
    }
    std::cout << "END OF TESTING EMPTY BINS" << std::endl;
    if(nproblast==0) isEmptyBinlast = false;
    
    incrlast++;
    halphadrE3.Reset();
  } // while(isEmptyBinLast)
  
  std::cout << "THE FINAL NUMBER OF BINS IN LAST R BIN IS : " << nRBinslast << std::endl;

  // ##################################################################################################################################

  // Need to recreate the final vector of dr bins outside of the loop !

  vector<float> vRBinsSTDlast; 
  Int_t q = 1 ; 
  for(Int_t i=0;i<nRBinsVal-1+nRBinslast;i++){
    if(i<vRBinsSTD.size()-1){
      vRBinsSTDlast.push_back(vRBinsSTD.at(i));
    }
    if(i>=vRBinsSTD.size()-1){      
      if(!doexp)
	vRBinsSTDlast.push_back(vRBinsSTD.at(vRBinsSTD.size()-2)+q*((vRBinsSTD.at(vRBinsSTD.size()-1)-vRBinsSTD.at(vRBinsSTD.size()-2))/nRBinslast));
      if(doexp)
	vRBinsSTDlast.push_back(vRBinsSTD.at(vRBinsSTD.size()-2)+(q*((vRBinsSTD.at(vRBinsSTD.size()-1)-vRBinsSTD.at(vRBinsSTD.size()-2))/nRBinslast))/(nRBinslast+1-q));
      q++;
    }
  }


  nRBinslast = nRBinslast + nRBins - 1 ; 
  const int nRBins2last  = nRBinslast + 1 ;
  const int nRBins3last  = 2*nRBins2last - 1 ; 
  Float_t vRBinslast[nRBins2last];
  Float_t mybinR2last[nRBins3last];

  for(Int_t i=0;i<nRBins2last;i++){
    vRBinslast[i] = vRBinsSTDlast.at(i) ;
    mybinR2last[i]= - vRBinsSTDlast.at(nRBins2last-i-1);
    if(i<nRBins2last-1) mybinR2last[i+nRBins2last] = vRBinsSTDlast.at(i+1) ;
  }

  // LAST R BINNING ######################################################################################################

  std::cout << "########################################################## " <<std::endl;
  std::cout << " 6. CHECK WITH LAST R BINNING ############################ " << std::endl;

  std::cout << "R BINNNING LAST " << std::endl;
  for(Int_t i=0;i<nRBins2last;i++){
    std::cout << i << "  " << vRBinslast[i] << std::endl;
  }

  //std::cout << "SYMMETRIC BINNNING LAST " << std::endl;
  //for(Int_t i=0;i<nRBins3last;i++){
  //  std::cout << i << "  " << mybinR2last[i] << std::endl;
  //}
    
  TH1F hdr4("hdr4","hdr4",nRBinslast,vRBinslast) ;
  hdr4.GetXaxis()->SetTitle("#delta r");

  TH1F halpha4("halpha4","halpha4",nAlphaBins,vAlphaBins) ;
  halpha4.GetXaxis()->SetTitle("#alpha");

  TH1F hdrE4("hdrE4","hdrE4",nRBinslast,vRBinslast) ;
  hdrE4.GetXaxis()->SetTitle("#delta r");

  TH1F halphaE4("halphaE4","halphaE4",nAlphaBins,vAlphaBins) ;
  halphaE4.GetXaxis()->SetTitle("#alpha");

  TMatrixD mEnergy4(nAlphaBins,nRBinslast);
  TMatrixD mxEnergy4(nAlphaBins,nRBinslast);
  TMatrixD myEnergy4(nAlphaBins,nRBinslast);

  TH2F halphadrE4 = fBinning(myfile,nxbins,xbins,nybins,ybins,
			     nAlphaBins+1,vAlphaBins,nRBinslast+1,vRBinslast, 
			     "Rfinal",selected_layer,doPart,
			     halpha4,hdr4,halphaE4,hdrE4,
			     mEnergy4,mxEnergy4,myEnergy4);

  // RECHECK #####################################################################################
  
  std::cout << "########################################################## " <<std::endl;
  std::cout << "7. FINAL CHECK :::::::::::: START TESTING EMPTY BINS" << std::endl;
  Int_t nprob = 0 ;
  for(Int_t i=1;i<halphadrE4.GetNbinsX()+1;i++){
    for(Int_t j=1;j<halphadrE4.GetNbinsY()+1;j++){
      if(halphadrE4.GetBinContent(i,j)<tolerance){
	std::cout << "PROBLEM WITH BIN (" << i << "," << j << ")" << std::endl;
      }
    }
  }
  
  return 0;

}

//##############################################################################################

vector<float> fVecBin(TH1F myh, int nBins, bool doR){

  vector<float> vbin;

  // to ensure that the outliers (dr>0.4) are not counted !
  float TotalValue     = 0;
  for(Int_t i=1;i<myh.GetNbinsX()+1;i++){
    TotalValue += myh.GetBinContent(i);
  }

  const int nBins2     = nBins + 1 ;
  float PerBinValue    = TotalValue/nBins ;
  
  float rotZach = 0 ; 
  if(!doR)
    rotZach = TMath::Pi()/8.;

  vector<float> mybincumul ;
  mybincumul.push_back(0) ; // initial value
  if(doR)  vbin.push_back(0) ;
  if(!doR) vbin.push_back(rotZach);

  Int_t q=0;
  Int_t count = 0 ;
  
  std::cout << "INTEGRAL : " << myh.GetEntries() << "  " << myh.Integral() << "  " << TotalValue << "  " << nBins << "  " << PerBinValue << "  " << myh.GetNbinsX()+1 << std::endl;

  for(Int_t g=1;g<myh.GetNbinsX()+1;g++){
    
  // we cumulate bins until we reach the max value
    if(mybincumul.at(q)<PerBinValue){
      mybincumul.at(q) = mybincumul.at(q) + myh.GetBinContent(g) ;
      count ++ ;
      if(g==myh.GetNbinsX()) std::cout << q << "  " << mybincumul.at(q) << std::endl;
    }
    // we have reached the max value 
    if(mybincumul.at(q)>=PerBinValue){
      // special case when one bin only already is > than the max value
      if(count==0){
	std::cout << "SPECIAL CASE #####################################################" << std::endl;
	mybincumul.at(q) = myh.GetBinContent(g) ;
      }
      std::cout << "BIN FILLED : " << q << "  " << mybincumul.at(q) << std::endl;
      q++;
      vbin.push_back(myh.GetBinCenter(g)) ;
      mybincumul.push_back(0) ;
      count = 0 ;
    }
  }
  if(doR){
    vbin.push_back(0.4) ;
  }
  if(!doR){
    vbin.push_back(2*TMath::Pi()+TMath::Pi()/8.) ; 
  }
  return vbin ;
}

