/*
  Copyright (C) 2002-2017 CERN for the benefit of the ATLAS collaboration
*/

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

int fGetIndex(float myvar,Float_t *vvarBin, Int_t nbins);

TH2F fBinning(string myfile, Int_t nxbins, Float_t *xbins, Int_t nybins, Float_t *ybins, Int_t nxbinsAlpha, Float_t *xbinsAlpha, Int_t nybinsR, Float_t *ybinsR, string mycase, float layer, string dopart, TH1F &halpha, TH1F &hdr, TH1F &halphaE, TH1F &hdrE, TMatrixD &mEnergy, TMatrixD &mxEnergy, TMatrixD &myEnergy);

TH2F fBinning(string myfile, Int_t nxbins, Float_t *xbins, Int_t nybins, Float_t *ybins, Int_t nxbinsAlpha, Float_t *xbinsAlpha, Int_t nybinsR, Float_t *ybinsR, string mycase, float layer, string doPart, TH1F &halpha, TH1F& hdr, TH1F &halphaE, TH1F &hdrE, TMatrixD &mEnergy, TMatrixD &mxEnergy, TMatrixD &myEnergy){

  TH1::SetDefaultSumw2(kTRUE);

  ostringstream os;
  os << layer ;

  string compname = os.str()+"_"+doPart+"_"+mycase+".eps" ;
  string compname2 = os.str()+"_"+doPart+"_"+mycase+".root" ;


  //std::cout << "BIN TEST : " << nxbinsAlpha << "  " << nybinsR << std::endl;
  //for(Int_t i=0;i<nxbinsAlpha;i++){
  //  std::cout << "BIN TEST PHI : " << xbinsAlpha[i] << std::endl;
  //}
  //for(Int_t j=0;j<nybinsR;j++){
  //std::cout << "BIN tEST R : " << ybinsR[j] << std::endl;
  //}

  //std::cout << "TEST MATRIX : " << mEnergy.GetNrows() << "  " << mEnergy.GetNcols() << std::endl;

  for(Int_t i=0;i<mEnergy.GetNrows();i++){
    for(Int_t j=0;j<mEnergy.GetNcols();j++){
      mEnergy(i,j)  = 0 ;
      mxEnergy(i,j) = 0 ;
      myEnergy(i,j) = 0 ;
    }
  }

  TFile* F       = TFile::Open(myfile.c_str());
  TTree *T       = (TTree*)F->Get("ISF_HitAnalysis/CaloHitAna");

  //TH2F* histovec[3];
 
  const int nentries = T->GetEntries();
  TH2F* halphadrEvec2[nentries];

  TH2F* hdhdf = new TH2F(("hdhdf_"+mycase).c_str()," ",nxbins-1,xbins,nybins-1,ybins) ; 
  hdhdf->Reset();
  hdhdf->GetXaxis()->SetTitle("#delta#eta");
  hdhdf->GetYaxis()->SetTitle("#delta#phi");
  
  TH2F* hdhdfTTC = new TH2F(("hdhdfTTC_"+mycase).c_str()," ",600,-3,3,600,-3,3) ;
  hdhdfTTC->Reset();
  hdhdfTTC->GetXaxis()->SetTitle("#delta#eta");
  hdhdfTTC->GetYaxis()->SetTitle("#delta#phi");

  TH2F* halphadr = new TH2F(("halphadr_"+mycase).c_str()," ",nxbinsAlpha-1,xbinsAlpha,nybinsR-1,ybinsR) ;
  halphadr->Reset();
  halphadr->GetXaxis()->SetTitle("#alpha");
  halphadr->GetYaxis()->SetTitle("#delta r");
  TH2F* halphadr_dummy2 = new TH2F("halphadr_dummy2","halphadr_dummy2",nybins-1,ybins,nybins-1,ybins) ;
  halphadr_dummy2->Reset();

  halpha.Reset();
  halpha.SetName(("halpha_"+mycase).c_str());
  halpha.GetXaxis()->SetTitle("#alpha");
  
  TH1F* halphaEvec[nentries];
  TH1F* hdrEvec[nentries];
  TH1F* hdrEAllvec[nentries];

  hdr.Reset();
  hdr.GetXaxis()->SetTitle("dr");
  hdr.SetName(("hdr_"+mycase).c_str());

  halphaE.Reset();
  halphaE.SetName(("halphaE_"+mycase).c_str());
  halphaE.GetXaxis()->SetTitle("#alpha");

  hdrE.Reset();
  hdrE.GetXaxis()->SetTitle("dr");
  hdrE.SetName(("hdrE_"+mycase).c_str());

  TH1F* hlayer = new TH1F(("hlayer_"+mycase).c_str(),"hlayer",24,-0.5,23.5) ;
  hlayer->Reset();
  hlayer->GetXaxis()->SetTitle("layer ID");
  
  TH1F hdrAll("hdrAll","hdrAll",40,0,4) ;
  hdrAll.Reset();
  hdrAll.SetName(("hdrAll_"+mycase).c_str());
  hdrAll.GetXaxis()->SetTitle("dr");

  TH1F hdrEAll("hdrEAll","hdrEAll",40,0,4) ;
  hdrEAll.Reset();
  hdrEAll.SetName(("hdrEAll_"+mycase).c_str());
  hdrEAll.GetXaxis()->SetTitle("dr");


  // BRANCHES #####################################

  //std::cout << "START BRANCHES" << std::endl;

  std::vector<float>* HitX = 0 ;
  T->SetBranchAddress("HitX",&HitX);

  std::vector<float>* HitY = 0;
  T->SetBranchAddress("HitY",&HitY);

  std::vector<float>* HitZ = 0;
  T->SetBranchAddress("HitZ",&HitZ);

  std::vector<float>* HitE = 0;
  T->SetBranchAddress("HitE",&HitE);

  std::vector<float>* HitSampling =0;
  T->SetBranchAddress("HitSampling", &HitSampling);

  vector< vector<double> >* TTC_entrance_eta = 0 ;
  T->SetBranchAddress("TTC_entrance_eta", &TTC_entrance_eta);

  vector< vector<double> >* TTC_entrance_phi = 0 ;
  T->SetBranchAddress("TTC_entrance_phi", &TTC_entrance_phi);

  vector< vector<double> >* TTC_entrance_r = 0 ;
  T->SetBranchAddress("TTC_entrance_r", &TTC_entrance_r);

  vector< vector<double> >* TTC_back_phi = 0 ;
  T->SetBranchAddress("TTC_back_phi", &TTC_back_phi);

  std::vector<float>* TruthPx = 0;
  T->SetBranchAddress("TruthPx",&TruthPx);

  std::vector<float>* TruthPy = 0;
  T->SetBranchAddress("TruthPy",&TruthPy);
  
  std::vector<float>* TruthPz = 0;
  T->SetBranchAddress("TruthPz",&TruthPz);

  std::vector<float>* TruthE = 0;
  T->SetBranchAddress("TruthE",&TruthE);

  std::vector<float>* HitCellIdentifier = 0;
  T->SetBranchAddress("HitCellIdentifier",&HitCellIdentifier);

  // Loop on events ###############################################

  float TotalE         = 0 ;

  float Shiftx = 0 ; 
  float Shifty = 0 ; 

  
  for(Int_t i=0;i<T->GetEntries();i++){
    
    T->GetEntry(i);
    
    // One energy histo per event
    char *histname = new char[10];
    sprintf(histname, "halphadrEvec2_%d",i);
    halphadrEvec2[i] = new TH2F(histname,"",nxbinsAlpha-1,xbinsAlpha,nybinsR-1,ybinsR) ;
    
    char *histname2 = new char[10];
    sprintf(histname2, "halphaEvec_%d",i);
    halphaEvec[i] = new TH1F(histname2,"",nxbinsAlpha-1,xbinsAlpha) ;

    char *histname3 = new char[10];
    sprintf(histname3, "hdrEvec_%d",i);
    hdrEvec[i] = new TH1F(histname3,"",nybinsR-1,ybinsR) ;

    char *histname4 = new char[10];
    sprintf(histname4, "hdrEAllvec_%d",i);
    hdrEAllvec[i] = new TH1F(histname4,"",40,0,4) ;



    // Loop on hits
    
    for(Int_t j=0;j<HitX->size();j++){
      
      TotalE += HitE->at(j) ;
      
      // layer 
      float layer_id = HitSampling->at(j) ;
      hlayer->Fill(layer_id,HitE->at(j));
      if(layer_id != layer)
	continue ;
      
      TVector3 * pos = new TVector3(HitX->at(j), HitY->at(j), HitZ->at(j)) ;
      float eta      = pos->PseudoRapidity() ;
      float phi      = pos->Phi() ;
      float r        = TMath::Sqrt(phi*phi + eta*eta) ;
      float x        = HitX->at(j);
      float y        = HitY->at(j);
      float energy   = HitE->at(j);
      
      // TTC quantities
      
      float eta_correction   = (TTC_entrance_eta->at(0)).at(layer_id) ;
      float phi_correction   = (TTC_entrance_phi->at(0)).at(layer_id) ;
      
      if(eta_correction < -900){
	eta_correction = 0 ;
	std::cout << "no valid eta_correction found" << std::endl;
      }
      if(phi_correction < -900){
	phi_correction = 0 ;
	std::cout << "no valid phi_correction found" << std::endl;
      }
      
      // Delta values
      
      float d_eta = eta - eta_correction ;
      
      float myy = (TTC_entrance_r->at(0)).at(layer_id) * TMath::Sin((TTC_entrance_phi->at(0)).at(layer_id));
      float myx = (TTC_entrance_r->at(0)).at(layer_id) * TMath::Cos((TTC_entrance_phi->at(0)).at(layer_id));
      TVector2 * myv2= new TVector2(myx,myy);
      TVector2 * myv3= new TVector2(pos->X(),pos->Y());
      float d_phi = myv3->DeltaPhi(*myv2) ;
      
      float d_r   = TMath::Sqrt(d_phi*d_phi + d_eta*d_eta);
      
      // Truth quantities
      
      TLorentzVector lv ;
      lv.SetPxPyPzE(TruthPx->at(0),TruthPy->at(0),TruthPz->at(0),TruthE->at(0));
      float TruthPhi = lv.Phi();
      TVector3 * myv = new TVector3(lv.X(),lv.Y(),lv.Z()) ;
      
      // Alpha angle 
      
      float alpha = TMath::ATan2(d_phi,d_eta); 
      // to change from [-pi,pi] to [0,2*pi]
      if(alpha<0)
	alpha = 2*TMath::Pi()+alpha ;
      
      // to rotate by pi/8
      //alpha = alpha + (TMath::Pi()/8.) ;
      //if(alpha>2*TMath::Pi())
      //alpha = alpha - 2*TMath::Pi() ;
      if(alpha<(TMath::Pi()/8.))
	alpha = 2*TMath::Pi()+alpha ; 

      // Weighted energy dr and alpha 
      
      int myXindex = fGetIndex(alpha,xbinsAlpha,nxbinsAlpha);
      int myYindex = fGetIndex(d_r,ybinsR,nybinsR);
      //      std::cout << "TEST MATRIX INDEX : " << myXindex << "  " << myYindex << std::endl;
      
      if(d_r<=0.4 && d_r>=0){
	mEnergy(myXindex,myYindex)  += energy ;
        mxEnergy(myXindex,myYindex) += energy*alpha ;
	myEnergy(myXindex,myYindex) += energy*d_r ;
      }
      
      // Shift Calculation 
      Shiftx += energy*d_r*cos(alpha);
      Shifty += energy*d_r*sin(alpha);


      // Fill histograms

      hdhdf->Fill(d_eta,d_phi);
      if(i==12) {
	hdhdfTTC->Fill(d_eta,d_phi);
	//if(d_r>0.4)
	//std::cout << "TTC : " << d_eta << "  " << d_phi << "  " << d_r << " " << energy << std::endl;
      }
      halphadr->Fill(alpha,d_r);
      halphadrEvec2[i]->Fill(alpha,d_r,energy);
      halpha.Fill(alpha);
      halphaEvec[i]->Fill(alpha,energy);
      hdrEvec[i]->Fill(d_r,energy);
      hdr.Fill(d_r);
      hdrAll.Fill(d_r);
      hdrEAllvec[i]->Fill(d_r,energy);
    } // loop on hits
  
    //if(i==12) std::cout << "TOTAL ENERGY : " << halphaEvec[i]->Integral() << "  " << hdrEAllvec[i]->Integral() << std::endl;

    // energy normalization per event
    if(halphadrEvec2[i]->Integral()>0){
      halphadrEvec2[i]->Scale(1./halphadrEvec2[i]->Integral());
    }

    if(halphaEvec[i]->Integral()>0){
      halphaEvec[i]->Scale(1./halphaEvec[i]->Integral());
    }

    if(hdrEvec[i]->Integral()>0){
      hdrEvec[i]->Scale(1./hdrEvec[i]->Integral());
    }

    if(hdrEAllvec[i]->Integral()>0){
      hdrEAllvec[i]->Scale(1./hdrEAllvec[i]->Integral());
    }

    
  } // loop on events

  std::cout << "FINAL MATRIX" << std::endl;
  
  for(Int_t i=0;i<mEnergy.GetNrows();i++){
    for(Int_t j=0;j<mEnergy.GetNcols();j++){
      mxEnergy(i,j) = mxEnergy(i,j)/mEnergy(i,j);
      myEnergy(i,j) = myEnergy(i,j)/mEnergy(i,j);
    }
  }

  // Final shift 

  std::cout << "THE SHIFT IN ENERGY IS : " << Shiftx/TotalE << "  " << Shifty/TotalE << std::endl;


  // total energy density 
  TH2F halphadrETot("halphadrETot","halphadrETot",nxbinsAlpha-1,xbinsAlpha,nybinsR-1,ybinsR) ;
  halphadrETot.GetXaxis()->SetTitle("#alpha");
  halphadrETot.GetYaxis()->SetTitle("#delta r");
  
  TH2F halphadrETotLN("halphadrETotLN","halphadrETotLN",nxbinsAlpha-1,xbinsAlpha,nybinsR-1,ybinsR) ;
  halphadrETotLN.GetXaxis()->SetTitle("#alpha");
  halphadrETotLN.GetYaxis()->SetTitle("#delta r");

  for(Int_t i=0;i<T->GetEntries();i++){
    halphadrETot.Add(halphadrEvec2[i]);
    halphaE.Add(halphaEvec[i]);
    hdrE.Add(hdrEvec[i]);
    hdrEAll.Add(hdrEAllvec[i]);
  }

  for(Int_t i=0;i<halphadrETot.GetNbinsX()+1;i++){
    for(Int_t j=0;j<halphadrETot.GetNbinsY()+1;j++){
      halphadrETotLN.SetBinContent(i,j,TMath::Log(halphadrETot.GetBinContent(i,j)));
    }
  }
  
  // Matrices

  vector<float> vRBinWeighted;
  vector<float> vRBinCenterWeighted;
  vector<float> vAlphaBinWeighted;
  vector<float> vAlphaBinCenter;
  vector<float> vAlphaBinCenterWeighted;
  vector<float> vRBinCenter;
  vector<float> vRBinLowEdge;
  vector<float> vRBinHighEdge;
  vector<float> vAlphaBinLowEdge;
  vector<float> vAlphaBinHighEdge;
  vector<float> venergy;

  string outTree = "OUTPUT/AF2histos_layer"+compname2;
  TFile f2(outTree.c_str(),"recreate");
  string mytreename = "layer"+os.str()+"_"+doPart ;
  TTree T2(mytreename.c_str(),"NewBinning");

  TBranch *b0 =T2.Branch("alphaBinCenter",         &vAlphaBinCenter );
  TBranch *b1 =T2.Branch("drBinCenter",            &vRBinCenter );
  TBranch *b2 =T2.Branch("alphaBinCenterWeighted", &vAlphaBinCenterWeighted );
  TBranch *b3 =T2.Branch("drBinCenterWeighted",    &vRBinCenterWeighted );
  TBranch *b4 =T2.Branch("alphaBinLowEdge",        &vAlphaBinLowEdge );
  TBranch *b5 =T2.Branch("alphaBinHighEdge",       &vAlphaBinHighEdge );
  TBranch *b6 =T2.Branch("drBinLowEdge",           &vRBinLowEdge );
  TBranch *b7 =T2.Branch("drBinHighEdge",          &vRBinHighEdge );
  TBranch *b8 =T2.Branch("energy",                 &venergy );

  std::cout << "Alpha matrix " << std::endl;
  mxEnergy.Print();
  std::cout << "dr matrix" << std::endl;
  myEnergy.Print();

  Int_t totbin = nybinsR*nxbinsAlpha ;
  Double_t vRBinCenterWeighted2[totbin];
  Double_t vAlphaBinCenterWeighted2[totbin];
  Double_t vRBinCenter2[totbin];
  Double_t vAlphaBinCenter2[totbin];

  Double_t vRBinError[totbin];
  Double_t vAlphaBinError[totbin];

  Int_t t2=0;

  std::cout << "BEFORE LOOP ON ENERGY PLOT " << std::endl;

  for(Int_t i=1;i<halphadrETot.GetNbinsX()+1;i++){
    for(Int_t j=1;j<halphadrETot.GetNbinsY()+1;j++){
      vAlphaBinCenter.push_back(halphaE.GetBinCenter(i));
      vAlphaBinCenter2[t2] = halphaE.GetBinCenter(i) ;
      vRBinCenter.push_back(hdrE.GetBinCenter(j));
      vRBinCenter2[t2] = hdrE.GetBinCenter(j) ;
      venergy.push_back(halphadrETot.GetBinContent(i,j));
      vAlphaBinLowEdge.push_back(xbinsAlpha[i-1]);
      vAlphaBinHighEdge.push_back(xbinsAlpha[i]);
      vRBinLowEdge.push_back(ybinsR[j-1]);
      vRBinHighEdge.push_back(ybinsR[j]);
      //                                                                                                                                                                             
      vRBinCenterWeighted.push_back(myEnergy(i-1,j-1));
      vRBinCenterWeighted2[t2] = myEnergy(i-1,j-1) ;
      vRBinError[t2] = vRBinCenterWeighted2[t2] - vRBinCenter2[t2] ;
      vAlphaBinCenterWeighted.push_back(mxEnergy(i-1,j-1));
      vAlphaBinCenterWeighted2[t2] = mxEnergy(i-1,j-1);
      vAlphaBinError[t2] = vAlphaBinCenterWeighted2[t2] - vAlphaBinCenter2[t2] ;
      t2++;
    }
  }

  std::cout << "BEFORE POLAR PLOT" << std::endl;
  TGraphPolar* gr = new TGraphPolar(totbin,vAlphaBinCenter2,vRBinCenter2,vAlphaBinError,vRBinError);
  gr->SetTitle("Bin centre versus energy weighted centre");

  T2.Fill();

  gStyle->SetOptLogz(0);

  hdhdf->SetName(("detavsdphi_layer"+os.str()).c_str());
  halphadr->SetName(("alphabsdr_layer"+os.str()).c_str());
  halphadrETot.SetName(("alphavsdrE_layer"+os.str()).c_str());
  halphadrETotLN.SetName(("alphavsdrLNE_layer"+os.str()).c_str());

  hdhdf->SetTitle((doPart+", layer"+os.str()+", number of hits").c_str());
  halphadr->SetTitle((doPart+", layer"+os.str()+", number of hits").c_str());
  halphadrETot.SetTitle((doPart+", layer"+os.str()+", energy").c_str());
  halphadrETotLN.SetTitle((doPart+", layer"+os.str()+", energy").c_str());


  TCanvas c("c","dphi versus deta",10,10,800,600);
  hdhdf->Draw("COLZ");
  c.SetRightMargin(0.13);
  c.Update();
  c.SaveAs(("OUTPUT/dphivsdeta_layer"+compname).c_str());

  TCanvas c1("c1","dr versus alpha",10,10,900,800);
  c1.SetRightMargin(0.13);
  halphadr_dummy2->Draw("COLZ");
  halphadr->Draw("COLZ POL SAME");
  halphadr_dummy2->SetTitle((doPart+", layer"+os.str()+", number of hits").c_str());
  c1.Update();
  c1.SaveAs(("OUTPUT/alphavsdr_layer"+compname).c_str());

  gStyle->SetOptLogz(1);

  TCanvas c2("c2","dr versus alpha",10,10,900,800);
  halphadr_dummy2->SetTitle((doPart+", layer"+os.str()+", energy").c_str());
  halphadr_dummy2->Draw("COLZ");
  halphadrETot.Draw("COLZ POL SAME");
  c2.SetRightMargin(0.13);
  c2.Update();
  c2.SaveAs(("OUTPUT/alphavsdrETot_layer"+compname).c_str());

  gStyle->SetOptLogz(0);

  TCanvas c23("c23","dr versus alpha",10,10,900,800);
  halphadr_dummy2->SetTitle((doPart+", layer"+os.str()+", LN(energy)").c_str());
  halphadr_dummy2->Draw("COLZ");
  halphadrETotLN.Draw("COLZ POL SAME");
  c23.SetRightMargin(0.13);
  c23.Update();
  c23.SaveAs(("OUTPUT/alphavsdrLNETot_layer"+compname).c_str());

  TCanvas c12("c12","dr versus alpha",10,10,900,800);
  c12.SetRightMargin(0.13);
  halphadr_dummy2->GetYaxis()->SetRangeUser(-0.1,0.1); 
  halphadr_dummy2->GetXaxis()->SetRangeUser(-0.1,0.1);
  halphadr_dummy2->Draw("COLZ");
  halphadr->Draw("COLZ POL SAME");
  halphadr_dummy2->SetTitle((doPart+", layer"+os.str()+", number of hits").c_str());
  c12.Update();
  c12.SaveAs(("OUTPUT/alphavsdrZOOM_layer"+compname).c_str());

  TCanvas c122("c122","dr versus alpha",10,10,900,800);
  c122.SetRightMargin(0.13);
  halphadr_dummy2->GetYaxis()->SetRangeUser(-0.03,0.03);
  halphadr_dummy2->GetXaxis()->SetRangeUser(-0.03,0.03);
  halphadr_dummy2->Draw("COLZ");
  halphadr->Draw("COLZ POL SAME");
  halphadr_dummy2->SetTitle((doPart+", layer"+os.str()+", number of hits").c_str());
  c122.Update();
  c122.SaveAs(("OUTPUT/alphavsdrZOOM2_layer"+compname).c_str());


  gStyle->SetOptLogz(1);

  TCanvas c22("c22","dr versus alpha",10,10,900,800);
  halphadr_dummy2->SetTitle((doPart+", layer"+os.str()+", energy").c_str());
  halphadr_dummy2->GetYaxis()->SetRangeUser(-0.1,0.1);
  halphadr_dummy2->GetXaxis()->SetRangeUser(-0.1,0.1);
  halphadr_dummy2->Draw("COLZ");
  halphadrETot.Draw("COLZ POL SAME");
  c22.SetRightMargin(0.13);
  c22.Update();
  c22.SaveAs(("OUTPUT/alphavsdrETotZOOM_layer"+compname).c_str());

  TCanvas c222("c222","dr versus alpha",10,10,900,800);
  halphadr_dummy2->SetTitle((doPart+", layer"+os.str()+", energy").c_str());
  halphadr_dummy2->GetYaxis()->SetRangeUser(-0.03,0.03);
  halphadr_dummy2->GetXaxis()->SetRangeUser(-0.03,0.03);
  halphadr_dummy2->Draw("COLZ");
  halphadrETot.Draw("COLZ POL SAME");
  c222.SetRightMargin(0.13);
  c222.Update();
  c222.SaveAs(("OUTPUT/alphavsdrETotZOOM2_layer"+compname).c_str());


  gStyle->SetOptLogz(0);

  TCanvas c24("c24","dr versus alpha",10,10,900,800);
  halphadr_dummy2->SetTitle((doPart+", layer"+os.str()+", LN(energy)").c_str());
  halphadr_dummy2->Draw("COLZ");
  halphadr_dummy2->GetYaxis()->SetRangeUser(-0.1,0.1);
  halphadr_dummy2->GetXaxis()->SetRangeUser(-0.1,0.1);
  halphadrETotLN.Draw("COLZ POL SAME");
  c24.SetRightMargin(0.13);
  c24.Update();
  c24.SaveAs(("OUTPUT/alphavsdrLNETotZOOM_layer"+compname).c_str());

  TCanvas c242("c242","dr versus alpha",10,10,900,800);
  halphadr_dummy2->SetTitle((doPart+", layer"+os.str()+", LN(energy)").c_str());
  halphadr_dummy2->Draw("COLZ");
  halphadr_dummy2->GetYaxis()->SetRangeUser(-0.03,0.03);
  halphadr_dummy2->GetXaxis()->SetRangeUser(-0.03,0.03);
  halphadrETotLN.Draw("COLZ POL SAME");
  c242.SetRightMargin(0.13);
  c242.Update();
  c242.SaveAs(("OUTPUT/alphavsdrLNETotZOOM2_layer"+compname).c_str());


  TCanvas c3("c3","layers",10,10,800,600);
  hlayer->SetFillColor(2);
  hlayer->Draw("BAR1");
  TLine* myline = new TLine(-0.5,0.05,23.5,0.05);
  myline->SetLineColor(kBlue);
  myline->SetLineWidth(3);
  myline->Draw("same");
  c3.Update();
  c3.SaveAs(("OUTPUT/layers_"+compname).c_str());

  TCanvas c4("c4","dr",10,10,800,600);
  hdr.Draw("COLZ");
  c4.SetRightMargin(0.13);
  c4.Update();
  c4.SaveAs(("OUTPUT/dr_layer"+compname).c_str());

  gStyle->SetOptLogy(1);
  TCanvas c42("c42","dr",10,10,800,600);
  hdrAll.Draw("COLZ");
  c42.SetRightMargin(0.13);
  c42.Update();
  c42.SaveAs(("OUTPUT/drAll_layer"+compname).c_str());
  gStyle->SetOptLogy(0);

  std::cout << "hdr<0.4 : " << hdrAll.Integral(1,4);
  std::cout << "hdr>0.4 : " << hdrAll.Integral(5,40);


  TCanvas c5("c5","alpha",10,10,800,600);
  halpha.Draw("COLZ");
  c5.SetRightMargin(0.13);
  c5.Update();
  c5.SaveAs(("OUTPUT/alpha_layer"+compname).c_str());

  TCanvas c6("c6","drE",10,10,800,600);
  hdrE.Draw("COLZ");
  c6.SetRightMargin(0.13);
  c6.Update();
  c6.SaveAs(("OUTPUT/drE_layer"+compname).c_str());

  gStyle->SetOptLogy(1);
  TCanvas c62("c62","drEAll",10,10,800,600);
  hdrEAll.Draw("COLZ");
  c62.SetRightMargin(0.13);
  c62.Update();
  c62.SaveAs(("OUTPUT/drEAll_layer"+compname).c_str());
  gStyle->SetOptLogy(0);

  TCanvas c7("c7","alphaE",10,10,800,600);
  halphaE.Draw("COLZ");
  c7.SetRightMargin(0.13);
  c7.Update();
  c7.SaveAs(("OUTPUT/alphaE_layer"+compname).c_str());


  gStyle->SetOptLogz(0);

  TCanvas c8("c8","dphi versus deta",10,10,800,600);                                                                                     
  hdhdfTTC->Draw("COLZ");                                                                                                                  
  c8.SetRightMargin(0.13);                                                                                                                
  c8.Update();                                                                                                                            
  c8.SaveAs(("OUTPUT/dphivsdetaTTC_layer"+compname).c_str());                                                           


  TCanvas c19("c19","dr versus alpha",10,10,900,800);
  gr->SetMarkerStyle(20);
  gr->SetMarkerColor(4);
  gr->SetMarkerSize(1);
  gr->SetLineWidth(2);
  gr->SetLineColor(2);
  gr->Draw("PE");
  c19.SetRightMargin(0.13);
  c19.Update();
  gr->GetPolargram()->SetToRadian();
  c19.SaveAs(("OUTPUT/center_layer"+compname).c_str());
  gStyle->SetOptLogz(1);                                                                                                                                                             
  //TObjArray Hlist(0) ;
  //Hlist.Add(gr);
  //Hlist.Add(hdhdf);
  //Hlist.Add(hdr);                                                                                                                                                                   
  //Hlist.Add(hdrE);
  //Hlist.Add(halpha);
  //Hlist.Add(halphaE);
  //Hlist.Add(halphadr);
  //Hlist.Add(halphadrETot);
  //Hlist.Add(hlayer);
  //
  //Hlist.Write();
  T2.Write();
  f2.Close();
  T2.Reset();

  //histovec[0]->Add(hdhdf);
  //histovec[1]->Add(halphadr);
  //histovec[2]->Add(halphadrETot);

  F->Close();

  return halphadrETot ;//histovec;

 }

int fGetIndex(float myvar, Float_t *myvarbin, Int_t nbins){

  int mycoord = -1000 ;

  for(Int_t i=0;i<nbins-1;i++){
    if(myvar>=myvarbin[i] && myvar<=myvarbin[i+1]){
      mycoord = i ;
      break;
    }
  }
  return mycoord ;
}

