/*
  Copyright (C) 2002-2021 CERN for the benefit of the ATLAS collaboration
*/

#include "FastCaloSimAnalyzer/TFCSAnalyzerBase.h"
#include "DSIDConverter.h"
#include "TChain.h"
#include "TGraph.h"
#include <fstream>
#include <sstream>
#include <iostream>

void check_layers();
void get_relevantlayers_inputs(vector<double> &, TreeReader* read_inputTree);
void plot_allfractions();
void plot_fcal();

using namespace std;

void distributions()
{
 
 vector<int> energyval;
 energyval.push_back(65536);
 energyval.push_back(131072);
 energyval.push_back(262144);
 
 vector<string> eta;
 eta.push_back("155");
 
 vector<string> pdgid;
 //pdgid.push_back("211");
 //pdgid.push_back("22");
 pdgid.push_back("11");
 
 string sampleData = "../../FastCaloSimAnalyzer/python/inputSampleList.txt";
 DSIDConverter* mydsid=new DSIDConverter();
 mydsid->init("db.txt");

 for(unsigned int e=0;e<energyval.size();e++)
 {
 
 string energy=Form("%i",energyval[e]);
 
 for(int p=0;p<pdgid.size();p++)
 {
  for(int h=0;h<eta.size();h++)
  {
   cout<<"energy "<<energy<<" pdg "<<pdgid[p]<<" eta "<<eta[h]<<endl;
   
   string dsid_s=mydsid->find_dsid(pdgid[p],energy,eta[h],"0");
   int dsid=atoi(dsid_s.c_str());
   TFCSAnalyzerBase::SampleInfo sample;
   sample = TFCSAnalyzerBase::GetInfo(sampleData.c_str(), dsid);
   string input = sample.inputSample;
   TChain* mychain= new TChain("FCS_ParametrizationInput");
   mychain->Add(input.c_str());
   TreeReader* read_inputTree = new TreeReader();
   read_inputTree->SetTree(mychain);
   
   vector<TH1D*> histos;
   for(int l=0;l<24;l++)
   {
    TH1D* hist=new TH1D(Form("h_layer%i",l),Form("h_layer%i",l),200,-1000,2*energyval[e]);
    histos.push_back(hist);
   }
   TH1D* h_totalE=new TH1D("h_totalE","h_totalE",1000,0.8*energyval[e],1.2*energyval[e]);
   
   for(int event=0;event<read_inputTree->GetEntries();event++ )
   {
    read_inputTree->GetEntry(event);
    double totalE=read_inputTree->GetVariable("total_cell_energy");
    h_totalE->Fill(totalE);
    for(int l=0;l<24;l++)
    {
     double eval=read_inputTree->GetVariable(Form("cell_energy[%i]",l));
     histos[l]->Fill(eval);
    }
   }
   
   for(int l=0;l<24;l++)
   {
    TCanvas* can=new TCanvas("can","can",0,0,800,600);
    histos[l]->Draw();
    histos[l]->GetXaxis()->SetTitle(Form("#eta=%s, pdgid=%s, E=%s MeV, E in layer %i [MeV]",eta[h].c_str(),pdgid[p].c_str(),energy.c_str(),l));
    double max=histos[l]->GetBinContent(histos[l]->GetMaximumBin());
    histos[l]->GetYaxis()->SetRangeUser(0.1,max*1.5);
    can->SetLogy();
    if(l==0) can->Print(Form("efracstudy/distris/pdg%s_E%s_eta%s.pdf(",pdgid[p].c_str(),energy.c_str(),eta[h].c_str()));
    else     can->Print(Form("efracstudy/distris/pdg%s_E%s_eta%s.pdf",pdgid[p].c_str(),energy.c_str(),eta[h].c_str()));
    can->Close();
    delete can;
   }
   
   TCanvas* can=new TCanvas("can","can",0,0,800,600);
   h_totalE->Draw();
   TLegend* leg=new TLegend(0.6,0.8,0.95,0.9);
   leg->SetBorderSize(0);
   leg->SetFillStyle(0);
   string percent="%";
   leg->SetHeader(Form("Mean=%.1f (%.2f%s), RMS=%.1f",h_totalE->GetMean(),h_totalE->GetMean()/(double)energyval[e]*100.0,percent.c_str(),h_totalE->GetRMS()));
   leg->Draw();
   h_totalE->GetXaxis()->SetTitle(Form("#eta=%s, pdgid=%s, E=%s MeV, total E [MeV]",eta[h].c_str(),pdgid[p].c_str(),energy.c_str()));
   can->Print(Form("efracstudy/distris/pdg%s_E%s_eta%s.pdf)",pdgid[p].c_str(),energy.c_str(),eta[h].c_str()));
   can->Close();
   delete can;
   
   delete mychain;
   delete read_inputTree;
   
  }
 }
 
 }
 
}

void check_layers()
{
 
 vector<string> energy;
 energy.push_back("1024");
 /*
 energy.push_back("2048");
 energy.push_back("4096");
 energy.push_back("8192");
 energy.push_back("16384");
 energy.push_back("32768");
 energy.push_back("65536");
 energy.push_back("131072");
 energy.push_back("262144");
 */
 vector<int> eta;
 for(int h=250;h<350;h+=5)
  eta.push_back(h);
 
 string pdgid="211";
 cout<<"pdgid? (211,11,22) "<<endl;
 cin>>pdgid;
 
 string sampleData = "../../FastCaloSimAnalyzer/python/inputSampleList.txt";
 
 DSIDConverter* mydsid=new DSIDConverter();
 mydsid->init("db.txt");

 for(int e=0;e<energy.size();e++)
 {
  
  cout<<"now run energy "<<energy[e]<<endl;
  
  vector<TGraph*> graphs;
  for(int l=0;l<24;l++)
  {
   TGraph* g_efrac=new TGraph();
   g_efrac->SetName(Form("g_efrac_layer%i",l));
   graphs.push_back(g_efrac);
  }
  
  for(int h=0;h<eta.size();h++)
  {
   cout<<"now run energy "<<energy[e]<<" and eta "<<eta[h]<<endl;
   
   double etaval=(double)eta[h]/100.0+0.025;
   
   string dsid_s=mydsid->find_dsid(pdgid,energy[e],Form("%i",eta[h]),"0");
   int dsid=atoi(dsid_s.c_str());
   
   cout<<"dsid "<<dsid<<endl;
   
   TFCSAnalyzerBase::SampleInfo sample;
   sample = TFCSAnalyzerBase::GetInfo(sampleData.c_str(), dsid);
   string input = sample.inputSample;
   
   TChain* mychain= new TChain("FCS_ParametrizationInput");
   mychain->Add(input.c_str());
   
   TreeReader* read_inputTree = new TreeReader();
   read_inputTree->SetTree(mychain);
   
   vector<double> efrac;
   get_relevantlayers_inputs(efrac, read_inputTree);
   
   for(int l=0;l<24;l++)
    graphs[l]->SetPoint(h,etaval,efrac[l]);
   
   delete read_inputTree;
   delete mychain;
   
  } //for eta
 
  TFile *output=TFile::Open(Form("efracstudy/pdg%s_e%s.root",pdgid.c_str(),energy[e].c_str()),"RECREATE");
  for(int l=0;l<24;l++)
   output->Add(graphs[l]);
  output->Write();
  
  for(int l=0;l<24;l++)
   delete graphs[l];
  graphs.clear();
  
 } //for energy
 
 delete mydsid;
 
}

void check_truth()
{
 
 int dsid=433859;
 string sampleData = "../../FastCaloSimAnalyzer/python/inputSampleList.txt"; 
 TFCSAnalyzerBase::SampleInfo sample;
 sample = TFCSAnalyzerBase::GetInfo(sampleData.c_str(), dsid);
 string input = sample.inputSample;
 
 TChain* mychain= new TChain("FCS_ParametrizationInput");
 mychain->Add(input.c_str());
 TreeReader* read_inputTree = new TreeReader();
 read_inputTree->SetTree(mychain);
 
 for(int event=0;event<read_inputTree->GetEntries();event++ )
 {
  read_inputTree->GetEntry(event);
  double e=read_inputTree->GetVariable("TruthE");
  double px=read_inputTree->GetVariable("TruthPx");
  double py=read_inputTree->GetVariable("TruthPy");
  double pz=read_inputTree->GetVariable("TruthPz");
  TLorentzVector vec; vec.SetPxPyPzE(px,py,pz,e);
  double eta=vec.Eta();
  if(fabs(eta)>3 || fabs(eta)<2.95) cout<<"event "<<event<<" eta "<<eta<<endl;
 }
 
}


void get_relevantlayers_inputs(vector<double> &efrac, TreeReader* read_inputTree)
{
  
  int NLAYER=24;
  double eps=0.000000000000000000001;
  
  vector<double> sum_efraction;
  for(int l=0;l<NLAYER;l++)
   sum_efraction.push_back(0.0);
  
  int entries=read_inputTree->GetEntries();
  int good_events=0;
  for(int event=0;event<entries;event++ )
  {
    int event_ok=1;
    read_inputTree->GetEntry(event);
    double total_e=read_inputTree->GetVariable("total_cell_energy");
    if(total_e>eps)
    {
     for(int l=0;l<NLAYER;l++)
     {
      double eval=read_inputTree->GetVariable(Form("cell_energy[%i]",l));
      if(eval/total_e>1 || eval/total_e<0) event_ok=0;
     }
     if(event_ok)
     {
      good_events++;
      for(int l=0;l<NLAYER;l++)
      {
       double eval=read_inputTree->GetVariable(Form("cell_energy[%i]",l));
       sum_efraction[l] += eval/total_e;
      }
     }
    } //total E is good
  }
  
  for(int l=0;l<NLAYER;l++)
  {
   cout<<"layer "<<l<<" sum_efraction "<<sum_efraction[l]<<" entries "<<entries<<" good_events "<<good_events<<" ratio "<<sum_efraction[l]/(double)good_events<<endl;
   efrac.push_back(sum_efraction[l]/(double)good_events);
   if(sum_efraction[l]/(double)good_events>0.001) cout<<" ---> layer "<<l<<" is relevant"<<endl;
  }
 
}

void plot_allfractions()
{
 
 //for each energy and pdg, plot the fractions (24 per plot) -> in total 27 plots. too many. make a selection 3 energy points 1, 32, 262 -> 9 plots
 
 vector<string> energy;
 energy.push_back("1024");
 energy.push_back("32768");
 energy.push_back("262144");
 
 vector<string> pdgid;
 pdgid.push_back("211");
 pdgid.push_back("11");
 pdgid.push_back("22");
 
 string lname[24];
 lname[0]="PreB";
 lname[1]="EMB1";
 lname[2]="EMB2";
 lname[3]="EMB3";
 lname[4]="PreE";
 lname[5]="EME1";
 lname[6]="EME2";
 lname[7]="EME3";
 lname[8]="HEC0";
 lname[9]="HEC1";
 lname[10]="HEC2";
 lname[11]="HEC3";
 lname[12]="TileB0";
 lname[13]="TileB1";
 lname[14]="TileB2";
 lname[15]="TileGap1";
 lname[16]="TileGap2";
 lname[17]="TileGap3";
 lname[18]="TileExt0";
 lname[19]="TileExt1";
 lname[20]="TileExt2";
 lname[21]="FCAL0";
 lname[22]="FCAL1";
 lname[23]="FCAL2";

 int col[24];
 col[0]=TColor::GetColor("#e6194b");
 col[1]=TColor::GetColor("#3cb44b");
 col[2]=TColor::GetColor("#ffe119");
 col[3]=TColor::GetColor("#0082c8");
 col[4]=TColor::GetColor("#f58231");
 col[5]=TColor::GetColor("#911eb4");
 col[6]=TColor::GetColor("#46f0f0");
 col[7]=TColor::GetColor("#f032e6");
 col[8]=TColor::GetColor("#d2f53c");
 col[9]=TColor::GetColor("#fabebe");
 col[10]=TColor::GetColor("#008080");
 col[11]=TColor::GetColor("#e6beff");
 col[12]=TColor::GetColor("#aa6e28");
 col[13]=TColor::GetColor("#ADFF2F");
 col[14]=TColor::GetColor("#800000");
 col[15]=TColor::GetColor("#aaffc3");
 col[16]=TColor::GetColor("#808000");
 col[17]=TColor::GetColor("#ffd8b1");
 col[18]=TColor::GetColor("#000080");
 col[19]=TColor::GetColor("#808080");
 col[20]=TColor::GetColor("#B0E0E6");
 col[21]=TColor::GetColor("#00BFFF");
 col[22]=TColor::GetColor("#F4A460");
 col[23]=TColor::GetColor("#2F4F4F");
 
 for(unsigned int p=0;p<pdgid.size();p++)
 {
  TCanvas* can=new TCanvas("can","can",0,0,1000,600);
  can->Divide(2,2);
  TLegend* leg=new TLegend(0.1,0.2,0.85,0.85);
  leg->SetNColumns(2);
  leg->SetBorderSize(0);
  leg->SetFillStyle(0);
  for(unsigned int e=0;e<energy.size();e++)
  {
   TFile* infile=TFile::Open(Form("efracstudy/pdg%s_e%s.root",pdgid[p].c_str(),energy[e].c_str()));
   can->cd(e+1);
   TMultiGraph* mg=new TMultiGraph(); mg->SetName("mg");
   for(int l=0;l<24;l++)
   {
    TGraph* graph=(TGraph*)infile->Get(Form("g_efrac_layer%i",l)); graph->SetName(Form("g_efrac_layer%i",l));
    graph->SetLineColor(col[l]);
    graph->SetLineWidth(1);
    if(l%2) graph->SetLineStyle(2);
    mg->Add(graph);
    if(e==0) leg->AddEntry(graph,Form("Layer %i (%s)",l,lname[l].c_str()),"l");
   }
   mg->Draw("al");
   mg->GetXaxis()->SetTitle(Form("#bf{PDGID=%s, E=%s MeV}       |#eta|",pdgid[p].c_str(),energy[e].c_str()));
   mg->GetYaxis()->SetTitle("Average energy fraction in layer");
  }
  can->cd(4);
  leg->Draw();
  can->Print(Form("efracstudy/plots/fractions_%s.pdf",pdgid[p].c_str()));
  
 }
 
}

 //plot the 3 fcal layers for every energy and every pdg -> 27 plots

void plot_fcal()
{
 
 vector<string> energy;
 energy.push_back("1024");
 energy.push_back("2048");
 energy.push_back("4096");
 energy.push_back("8192");
 energy.push_back("16384");
 energy.push_back("32768");
 energy.push_back("65536");
 energy.push_back("131072");
 energy.push_back("262144");
 vector<int> eta;
 for(int h=0;h<350;h+=5)
  eta.push_back(h);
 
 vector<string> pdgid;
 pdgid.push_back("211");
 pdgid.push_back("11");
 pdgid.push_back("22");
 
 string lname[24];
 lname[21]="FCAL0";
 lname[22]="FCAL1";
 lname[23]="FCAL2";

 int col[24];
 col[21]=TColor::GetColor("#00BFFF");
 col[22]=TColor::GetColor("#F4A460");
 col[23]=TColor::GetColor("#2F4F4F");
 
 for(unsigned int p=0;p<pdgid.size();p++)
 {
  for(unsigned int e=0;e<energy.size();e++)
  {
   TFile* infile=TFile::Open(Form("efracstudy/pdg%s_e%s.root",pdgid[p].c_str(),energy[e].c_str()));
   TMultiGraph* mg=new TMultiGraph(); mg->SetName("mg");
   TLegend* leg=new TLegend(0.2,0.7,0.4,0.9);
   for(int l=21;l<24;l++)
   {
    TGraph* graph=(TGraph*)infile->Get(Form("g_efrac_layer%i",l)); graph->SetName(Form("g_efrac_layer%i",l));
    graph->SetLineColor(col[l]);
    graph->SetLineWidth(2);
    mg->Add(graph);
    leg->AddEntry(graph,Form("Layer %i (%s)",l,lname[l].c_str()),"l");
   }
   leg->SetBorderSize(0);
   leg->SetFillStyle(0);
   TCanvas* can=new TCanvas("can","can",0,0,800,600);
   mg->Draw("al");
   mg->GetYaxis()->SetRangeUser(0.0001,2.0);
   mg->GetXaxis()->SetTitle(Form("#bf{PDGID=%s, E=%s MeV}       |#eta|",pdgid[p].c_str(),energy[e].c_str()));
   mg->GetYaxis()->SetTitle("Average energy fraction in layer");
   leg->Draw();
   can->SetLogy();
   can->SetGridy();
   can->SetGridx();
   if(e==0) can->Print(Form("efracstudy/plots/FCALfractions_%s.pdf(",pdgid[p].c_str()));
   else if(e==energy.size()-1) can->Print(Form("efracstudy/plots/FCALfractions_%s.pdf)",pdgid[p].c_str()));
   else can->Print(Form("efracstudy/plots/FCALfractions_%s.pdf",pdgid[p].c_str()));
   can->Close();
   delete can;
  } // for energy
 }
 
 
}