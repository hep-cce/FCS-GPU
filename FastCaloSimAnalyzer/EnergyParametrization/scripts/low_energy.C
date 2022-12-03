/*
  Copyright (C) 2002-2023 CERN for the benefit of the ATLAS collaboration
*/

#include "FastCaloSimAnalyzer/TFCSAnalyzerBase.h"
#include "DSIDConverter.h"
#include "TChain.h"
#include "TFCSMakeFirstPCA.h"
#include "TFCSApplyFirstPCA.h"
#include "secondPCA.h"
#include "TFile.h"
#include "TH2I.h"
#include "TH1D.h"
#include "TRandom3.h"
#include "TCanvas.h"
#include "TLegend.h"
#include "ISF_FastCaloSimEvent/TFCSPCAEnergyParametrization.h"
#include "ISF_FastCaloSimEvent/TFCSTruthState.h"
#include "ISF_FastCaloSimEvent/TFCSExtrapolationState.h"
#include "ISF_FastCaloSimEvent/TFCSSimulationState.h"
#include "EnergyParametrizationValidation.h"
#include "TFCSEnergyParametrizationPCABinCalculator.h"
#include <fstream>
#include <sstream>
#include <iostream>

void low_energy();
void low_energy(int);


void low_energy()
{
 low_energy(430080);
}

void low_energy(int dsid)
{
 
 /*
 //64 MeV sample:
 
 central photons 430004: lowest E: -1 MeV  E=0: 7.1%,  E<-25MeV: 0%
 fcal photon 430080: lowest E: -843 MeV,   E=0: 90.63%,  E<-25MeV: 2.96%
 
 central el 431704: lowest E: -13 MeV   E=0: 37.57%,  E<-25MeV: 0%
 fcal el 431780: lowest E: -804 MeV     E=0: 93.02%,  E<-25MeV: 1.35%
 
 central pion 433404: lowest E: -22 MeV   E=0: 76.65%,  E<-25MeV: 0%
 fcal pion 433480: lowest E: -578 MeV     E=0: 82.66%,  E<-25MeV: 0.35%
 
 example fcal pion
 E=-577 MeV, all energy in Layer21
 E=-46 MeV, 7 MeV  in Layer7, -53 MeV in Layer8
 So mostly all goes in 1 layer, sometimes in 2, with the negative in 1.
 */
 
 string sampleData = "../../FastCaloSimAnalyzer/python/inputSampleList.txt";
 string topDir = "./output/";
 string version = "ver01";
 
 TFCSAnalyzerBase::SampleInfo sample;
 sample = TFCSAnalyzerBase::GetInfo(sampleData.c_str(), dsid);
 string input = sample.inputSample;
 string baselabel=Form("ds%i",dsid);
 
 system(("mkdir -p " + topDir).c_str());
 TString inputSample(Form("%s", input.c_str()));
 
 cout<<"*** Preparing to run on "<<inputSample <<" ***"<<endl;
 TChain* mychain= new TChain("FCS_ParametrizationInput");
 mychain->Add(inputSample);
 cout<<"TChain entries: "<<mychain->GetEntries()<<endl;
 
 TreeReader* read_inputTree = new TreeReader();
 read_inputTree->SetTree(mychain);
 
 vector<int> layer_number;
  
  int NLAYER=25;
  
  TH1D* h_totalE=new TH1D("h_totalE","h_totalE",1000,-100,200);
  TH1D* h_e[25];
  TH1D* h_efrac[25];
  double emin_total=10000.0;
  double emin[25];
  for(int l=0;l<NLAYER;l++)
  {
   h_e[l]=new TH1D(Form("h_e%i",l),Form("h_e%i",l),100,-100,200);
   h_efrac[l]=new TH1D(Form("h_efrac%i",l),Form("h_e%i",l),100,-2,3);
   emin[l]=100000.0;
  }
  
  int below0=0;
  int below25=0;
  int below1=0;
  int beloweps1=0;
  int beloweps2=0;
  int beloweps3=0;
  int beloweps4=0;
  
  int nentries=read_inputTree->GetEntries();
  for(int event=0;event<read_inputTree->GetEntries();event++ )
  {
   int check=0;
   read_inputTree->GetEntry(event);
   int event_ok=1;
   bool pass_eta=0;
   double total_e=read_inputTree->GetVariable("total_cell_energy");
   h_totalE->Fill(total_e);
   //cout<<"event "<<event<<" totalE "<<total_e<<endl;
   if(total_e<emin_total) emin_total=total_e;
   if(total_e<0)  below0++;
   if(total_e<-25)
   {
   	below25++;
   	check=1;
   }
   if(total_e<1.0 && total_e>=0) below1++;
   if(total_e<0.01 && total_e>=0) beloweps1++;
   if(total_e<0.0001 && total_e>=0) beloweps2++;
   if(total_e<0.000001 && total_e>=0) beloweps3++;
   if(total_e<0.00000001 && total_e>=0) beloweps4++;
   if(check) cout<<"totalE "<<total_e<<endl;
   for(int l=0;l<NLAYER;l++)
   {
    double e=read_inputTree->GetVariable(Form("cell_energy[%d]",l));
    double efraction = read_inputTree->GetVariable(Form("cell_energy[%d]",l))/total_e;
    h_e[l]->Fill(e);
    h_efrac[l]->Fill(efraction);
    if(e<emin[l]) emin[l]=e;
    if(check) cout<<"  l "<<l<<" e "<<e<<" fraction "<<efraction<<endl;
   }
   if(event%2000==0) cout<<event<<" from "<<read_inputTree->GetEntries()<<" done"<<endl;
  }
 
 cout<<"EMin_total "<<emin_total<<endl;
 for(int l=0;l<NLAYER;l++)
  cout<<"layer "<<l<<" emin "<<emin[l]<<endl;
 
 cout<<"Fraction of events with total 0<= E < 1: "<<(double)below1/(double)nentries*100.0<<"%"<<endl;
 cout<<"Fraction of events with total 0<= E < 0.01: "<<(double)beloweps1/(double)nentries*100.0<<"%"<<endl;
 cout<<"Fraction of events with total 0<= E < 0.0001: "<<(double)beloweps2/(double)nentries*100.0<<"%"<<endl;
 cout<<"Fraction of events with total 0<= E < 0.000001: "<<(double)beloweps3/(double)nentries*100.0<<"%"<<endl;
 cout<<"Fraction of events with total 0<= E < 0.00000001: "<<(double)beloweps4/(double)nentries*100.0<<"%"<<endl;
 cout<<"Fraction of events with negative total E: "<<(double)below0/(double)nentries*100.0<<"%"<<endl;
 cout<<"Fraction of events with total E < 25 MeV: "<<(double)below25/(double)nentries*100.0<<"%"<<endl;
 
 TFile* file=TFile::Open("test.root","RECREATE");
 file->Add(h_totalE);
 file->Write();
 
}

