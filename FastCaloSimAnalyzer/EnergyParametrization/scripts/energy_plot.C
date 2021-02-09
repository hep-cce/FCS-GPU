/*
  Copyright (C) 2002-2021 CERN for the benefit of the ATLAS collaboration
*/

void energy_plot();
void get_energy(int,double &energy_val_g4, double &energy_val_fast, double &energy_rms_g4, double &energy_rms_fast);
void get_totalEhisto(int,TH1D* h_input, TH1D* h_output);

void energy_plot_fulleta()
{
 
 string name="photon_131GeV"; //pion_131GeV, el_131GeV, photon_131GeV
 double e_true=131072.0;
 
 vector<int> dsid;
 vector<double> eta; 
 
 int count=0;
 ifstream file(Form("list_%s.txt",name.c_str()));
 if(!file) cout<<"file not found :O"<<endl;
 string line;
 while(getline(file, line))
 {
  stringstream linestream(line);
  string dsname;
  linestream >> dsname;
  cout<<"found dsid "<<dsname<<endl;
  dsid.push_back(atoi(dsname.c_str()));
  eta.push_back(0.025+0.05*count);
  count++;
 } // while file

 string version="ver01";
 
 TGraphErrors* g_g4=new TGraphErrors(); g_g4->SetName("g_g4");
 TGraphErrors* g_fast=new TGraphErrors(); g_fast->SetName("g_fast");
 
 TGraphErrors* g_g4_ratio=new TGraphErrors(); g_g4_ratio->SetName("g_g4_ratio");
 TGraphErrors* g_fast_ratio=new TGraphErrors(); g_fast_ratio->SetName("g_fast_ratio");
 
 for(int s=0;s<dsid.size();s++)
 {
  cout<<"now do "<<dsid[s]<<endl;
  
  double energy_g4,rms_g4,energy_fast,rms_fast;
  
  get_energy(dsid[s],energy_g4,energy_fast,rms_g4,rms_fast);
  g_g4->SetPoint(s,eta[s],energy_g4);
  g_g4->SetPointError(s,0.000001,rms_g4);
  
  g_fast->SetPoint(s,eta[s],energy_fast);
  g_fast->SetPointError(s,0.000001,rms_fast);
  
  g_g4_ratio->SetPoint(s,eta[s],energy_g4/e_true);
  g_g4_ratio->SetPointError(s,0.025,rms_g4/e_true);
  
  g_fast_ratio->SetPoint(s,eta[s],energy_fast/e_true);
  g_fast_ratio->SetPointError(s,0.025,rms_fast/e_true);

  cout<<"done with "<<dsid[s]<<endl;
 }
 
 TCanvas* can=new TCanvas("can","can",0,0,1200,800);
 can->Range(0,0,1,1);
 TPad* pad1=new TPad("pad1","pad1",0.0,0.35,1.0,1.0);
 pad1->SetBottomMargin(0.027);
 pad1->SetLeftMargin(0.12);
 pad1->SetRightMargin(0.03);
 pad1->Draw();
 TPad* pad2=new TPad("pad2","pad2",0.0,0.0,1.0,0.35);
 pad2->SetTopMargin(0.027);
 pad2->SetBottomMargin(0.28);
 pad2->SetLeftMargin(0.12);
 pad2->SetRightMargin(0.03);
 pad2->Draw();
 
 pad1->cd();
 g_g4_ratio->Draw("alp");
 g_g4_ratio->GetXaxis()->SetRangeUser(0,3.5);
 g_g4_ratio->GetXaxis()->SetLabelSize(0.00001);
 g_g4_ratio->GetYaxis()->SetTitle("Simulated energy/true energy");
 g_g4_ratio->GetYaxis()->SetTitleOffset(0.8);
 g_g4_ratio->SetMarkerSize(0.8);
 g_fast_ratio->SetMarkerSize(0.8);
 g_fast_ratio->SetMarkerColor(2); g_fast_ratio->SetLineStyle(2); g_fast_ratio->SetLineColor(2); g_fast_ratio->SetMarkerStyle(4);
 g_fast_ratio->Draw("lpsame");
 TLegend* leg2=new TLegend(0.15,0.1,0.4,0.25);
 leg2->SetBorderSize(0);
 leg2->AddEntry(g_g4_ratio,"#bf{Geant4 (mean#pmrms)}","lpe");
 leg2->AddEntry(g_fast_ratio,"#bf{FastSim (mean#pmrms)}","lpe");
 leg2->Draw();
 
 pad2->cd();
 TGraphErrors* g_ratio=new TGraphErrors();
 for(int i=0;i<g_g4_ratio->GetN();i++)
 {
  double eta, g4_mean,fast_mean;
  g_g4_ratio->GetPoint(i,eta,g4_mean);
  g_fast_ratio->GetPoint(i,eta,fast_mean);
  double ratio=g4_mean/fast_mean;
  //double ratio=(fast_mean-g4_mean);
  g_ratio->SetPoint(i,eta,ratio);
  g_ratio->SetPointError(i,0.025,0.00001);
 }
 g_ratio->Draw("ape");
 g_ratio->GetXaxis()->SetRangeUser(0,3.5);
 g_ratio->GetXaxis()->SetTitle(Form("#bf{%s}                |#eta_{Sample}|",name.c_str()));
 g_ratio->GetYaxis()->SetTitle("G4 / Fast");
 g_ratio->GetYaxis()->SetLabelSize(0.07);
 g_ratio->GetYaxis()->SetTitleSize(0.08);
 g_ratio->GetYaxis()->SetTitleOffset(0.7);
 g_ratio->GetXaxis()->SetTitleSize(0.09);
 g_ratio->GetXaxis()->SetTitleOffset(1.2);
 g_ratio->GetXaxis()->SetLabelSize(0.08);
 pad1->SetGridx();
 pad1->SetGridy();
 pad2->SetGridx();
 pad2->SetGridy();
 can->Print(Form("plots/fulleta_%s.pdf",name.c_str()));
 
 /*
 //now plot the totalE
 for(int s=0;s<dsid.size();s++)
 {
  cout<<"now do "<<dsid[s]<<endl;
  TH1D* h_input; TH1D* h_output;
  
  TFile* file=TFile::Open(Form("output/ds%i.eparavalidation.%s.root",dsid[s],version.c_str()));
  h_input =(TH1D*)file->Get(Form("h_input_zoom_totalE"));  h_input->SetName("h_input");
  h_output=(TH1D*)file->Get(Form("h_output_zoom_totalE")); h_output->SetName("h_output");
  
  h_output->Scale(h_input->Integral()/h_output->Integral());
  h_output->SetLineColor(2); h_output->SetMarkerColor(2); h_output->SetLineStyle(2); h_output->SetMarkerStyle(4);
  
  TCanvas* c1=new TCanvas("c1","c1",0,0,1300,600);
  c1->Divide(2);
  c1->cd(1);
  c1->cd(1)->SetRightMargin(0.1);
  c1->cd(1)->SetLeftMargin(0.1);
  double ymax=h_input->GetBinContent(h_input->GetMaximumBin());
  if(h_output->GetBinContent(h_output->GetMaximumBin())>ymax) ymax=h_output->GetBinContent(h_output->GetMaximumBin());
  ymax*=1.2;
  h_input->Draw("e");
  h_input->GetXaxis()->SetTitle(Form("pdg22 E=131GeV |#eta|=%.2f-%.2f      E [MeV]",eta[s]-0.025,eta[s]+0.025));
  h_input->GetYaxis()->SetRangeUser(0.1,ymax);
  h_output->Draw("esame");
  TLegend* leg=new TLegend(0.2,0.75,0.4,0.9);
  leg->SetFillStyle(0);
  leg->SetBorderSize(0);
  leg->AddEntry(h_input,"G4","lpe");
  leg->AddEntry(h_output,"FastSim","lpe");
  leg->Draw();
  c1->cd(2);
  c1->cd(2)->SetRightMargin(0.1);
  c1->cd(2)->SetLeftMargin(0.1);
  h_input->Draw("e");
  h_output->Draw("esame");
  c1->cd(2)->SetLogy();
  c1->Print(Form("plots/totalE_pdg22_131GeV_eta%i.pdf",(int)((eta[s]-0.025)*100.0)));
  c1->Close();
  delete c1;
  
  delete h_input; delete h_output;
  
 }
 */
 
}

void energy_plot_eta()
{
 
 system("mkdir -p plots");
 string type="pions";
 string energy="8GeV";
 double e_true;
 
 vector<int> dsid;
 vector<double> eta;
 eta.push_back(0.5);
 eta.push_back(1.0);
 eta.push_back(1.5);
 eta.push_back(2.0);
 eta.push_back(2.5);
 eta.push_back(3.0); 
 
 if(type=="pions" && energy=="8GeV")
 {
  e_true=8192.0;
  dsid.push_back(434110);
  dsid.push_back(434120);
  dsid.push_back(434130);
  dsid.push_back(434140);
  dsid.push_back(434150);
  dsid.push_back(434160);
 }
 
 if(type=="pions" && energy=="131GeV")
 {
  e_true=131072.0;
  dsid.push_back(434510);
  dsid.push_back(434520);
  dsid.push_back(434530);
  dsid.push_back(434540);
  dsid.push_back(434550);
  dsid.push_back(434560);
 }
 
 if(type=="photons" && energy=="8GeV")
 {
  e_true=8192.0;
  dsid.push_back(430710);
  dsid.push_back(430720);
  dsid.push_back(430730);
  dsid.push_back(430740);
  dsid.push_back(430750);
  dsid.push_back(430760);
 }
 
 if(type=="photons" && energy=="131GeV")
 {
  e_true=131072.0;
  dsid.push_back(431110);
  dsid.push_back(431120);
  dsid.push_back(431130);
  dsid.push_back(431140);
  dsid.push_back(431150);
  dsid.push_back(431160);
 }
 
 TGraphErrors* g_g4=new TGraphErrors(); g_g4->SetName("g_g4");
 TGraphErrors* g_fast=new TGraphErrors(); g_fast->SetName("g_fast");
 
 TGraphErrors* g_g4_ratio=new TGraphErrors(); g_g4_ratio->SetName("g_g4_ratio");
 TGraphErrors* g_fast_ratio=new TGraphErrors(); g_fast_ratio->SetName("g_fast_ratio");
 
 string version="ver01";
 
 for(int s=0;s<dsid.size();s++)
 {
 	cout<<"now do "<<dsid[s]<<endl;
 	
  double energy_g4,rms_g4,energy_fast,rms_fast;
  
  get_energy(dsid[s],energy_g4,energy_fast,rms_g4,rms_fast);
  g_g4->SetPoint(s,eta[s],energy_g4);
  g_g4->SetPointError(s,0.000001,rms_g4);
  
  g_fast->SetPoint(s,eta[s],energy_fast);
  g_fast->SetPointError(s,0.000001,rms_fast);
  
  g_g4_ratio->SetPoint(s,eta[s],energy_g4/e_true);
  g_g4_ratio->SetPointError(s,0.000001,rms_g4/e_true);
  
  g_fast_ratio->SetPoint(s,eta[s],energy_fast/e_true);
  g_fast_ratio->SetPointError(s,0.000001,rms_fast/e_true);

  cout<<"done with "<<dsid[s]<<endl;
 }
 
 TCanvas* can=new TCanvas("can","can",0,0,1200,600);
 can->Divide(2);
 can->cd(1);
 g_g4->Draw("alp");
 g_g4->GetXaxis()->SetTitle(Form("True eta #bf{%s E=%i MeV}",type.c_str(),(int)e_true));
 g_g4->GetYaxis()->SetTitle("Simulated Energy [MeV]");
 g_g4->GetYaxis()->SetTitleOffset(1.6);
 g_fast->SetMarkerColor(2); g_fast->SetLineStyle(2); g_fast->SetLineColor(2); g_fast->SetMarkerStyle(4);
 g_fast->Draw("lpsame");
 TLegend* leg=new TLegend(0.2,0.2,0.4,0.35);
 leg->SetBorderSize(0);
 leg->AddEntry(g_g4,"Geant4","lpe");
 leg->AddEntry(g_fast,"FastSim","lpe");
 leg->Draw();
 can->cd(1)->SetGridx();
 can->cd(1)->SetGridy();
 
 can->cd(2);
 g_g4_ratio->Draw("alp");
 g_g4_ratio->GetXaxis()->SetTitle(Form("True eta #bf{%s E=%i MeV}",type.c_str(),(int)e_true));
 g_g4_ratio->GetYaxis()->SetTitle("Simulated energy/true energy");
 g_fast_ratio->SetMarkerColor(2); g_fast_ratio->SetLineStyle(2); g_fast_ratio->SetLineColor(2); g_fast_ratio->SetMarkerStyle(4);
 g_fast_ratio->Draw("lpsame");
 TLegend* leg2=new TLegend(0.2,0.2,0.4,0.35);
 leg2->SetBorderSize(0);
 leg2->AddEntry(g_g4_ratio,"Geant4","lpe");
 leg2->AddEntry(g_fast_ratio,"FastSim","lpe");
 leg2->Draw();
 can->cd(2)->SetGridx();
 can->cd(2)->SetGridy();
 
 can->Print(Form("plots/energy_vs_eta_%s_E%i.pdf",type.c_str(),(int)e_true/1000));
 
}


void energy_plot()
{
 
 system("mkdir -p plots");
 
 string type="pions";
 
 vector<int> dsid;
 vector<string> descr;
 vector<double> e_true;
 
 if(type=="photons")
 {
  dsid.push_back(430404); descr.push_back("#gamma E=1 GeV 0.2<|#eta|<0.25"); e_true.push_back(1024);
  dsid.push_back(430504); descr.push_back("#gamma E=2 GeV 0.2<|#eta|<0.25"); e_true.push_back(2048);
  dsid.push_back(430604); descr.push_back("#gamma E=4 GeV 0.2<|#eta|<0.25"); e_true.push_back(4096);
  dsid.push_back(430704); descr.push_back("#gamma E=8 GeV 0.2<|#eta|<0.25"); e_true.push_back(8192);
  dsid.push_back(430804); descr.push_back("#gamma E=16 GeV 0.2<|#eta|<0.25"); e_true.push_back(16384);
  dsid.push_back(430904); descr.push_back("#gamma E=32 GeV 0.2<|#eta|<0.25"); e_true.push_back(32768);
  dsid.push_back(431004); descr.push_back("#gamma E=65 GeV 0.2<|#eta|<0.25"); e_true.push_back(65536);
  dsid.push_back(431104); descr.push_back("#gamma E=131 GeV 0.2<|#eta|<0.25"); e_true.push_back(131072);
  dsid.push_back(431204); descr.push_back("#gamma E=262 GeV 0.2<|#eta|<0.25"); e_true.push_back(262144);
 }
 
 string extra="_5x2pca";
 if(type=="pions")
 {
  dsid.push_back(433804); descr.push_back("#gamma E=1 GeV 0.2<|#eta|<0.25");     e_true.push_back(1024);
  dsid.push_back(433904); descr.push_back("#gamma E=2 GeV 0.2<|#eta|<0.25");     e_true.push_back(2048);
  dsid.push_back(434004); descr.push_back("#gamma E=4 GeV 0.2<|#eta|<0.25");     e_true.push_back(4096);
  dsid.push_back(434104); descr.push_back("#gamma E=8 GeV 0.2<|#eta|<0.25");     e_true.push_back(8192);
  dsid.push_back(434204); descr.push_back("#gamma E=16 GeV 0.2<|#eta|<0.25");   e_true.push_back(16384);
  dsid.push_back(434304); descr.push_back("#gamma E=32 GeV 0.2<|#eta|<0.25");   e_true.push_back(32768);
  dsid.push_back(434404); descr.push_back("#gamma E=65 GeV 0.2<|#eta|<0.25");   e_true.push_back(65536);
  dsid.push_back(434504); descr.push_back("#gamma E=131 GeV 0.2<|#eta|<0.25"); e_true.push_back(131072);
  dsid.push_back(434604); descr.push_back("#gamma E=262 GeV 0.2<|#eta|<0.25"); e_true.push_back(262144);
 }
 
 TGraphErrors* g_g4=new TGraphErrors(); g_g4->SetName("g_g4");
 TGraphErrors* g_fast=new TGraphErrors(); g_fast->SetName("g_fast");
 
 TGraphErrors* g_g4_ratio=new TGraphErrors(); g_g4_ratio->SetName("g_g4_ratio");
 TGraphErrors* g_fast_ratio=new TGraphErrors(); g_fast_ratio->SetName("g_fast_ratio");
 
 string version="ver01";
 
 for(int s=0;s<dsid.size();s++)
 {
 	cout<<"now do "<<dsid[s]<<endl;
 	
  double energy_g4,rms_g4,energy_fast,rms_fast;
  
  get_energy(dsid[s],energy_g4,energy_fast,rms_g4,rms_fast);
  g_g4->SetPoint(s,e_true[s],energy_g4);
  g_g4->SetPointError(s,0.000001,rms_g4);
  
  g_fast->SetPoint(s,e_true[s],energy_fast);
  g_fast->SetPointError(s,0.000001,rms_fast);
  
  g_g4_ratio->SetPoint(s,e_true[s],energy_g4/e_true[s]);
  g_g4_ratio->SetPointError(s,0.000001,rms_g4/e_true[s]);
  
  g_fast_ratio->SetPoint(s,e_true[s],energy_fast/e_true[s]);
  g_fast_ratio->SetPointError(s,0.000001,rms_fast/e_true[s]);

  cout<<"done with "<<dsid[s]<<endl;
 }
 
 cout<<"now make energy response plot"<<endl;
 
 TCanvas* can=new TCanvas("can","can",0,0,800,600);
 g_g4->Draw("alp");
 g_g4->GetXaxis()->SetTitle(Form("True energy [MeV] #bf{%s}",type.c_str()));
 g_g4->GetYaxis()->SetTitle("Simulated Energy [MeV]");
 g_fast->SetMarkerColor(2); g_fast->SetLineStyle(2); g_fast->SetLineColor(2); g_fast->SetMarkerStyle(4);
 g_fast->Draw("lpsame");
 TLegend* leg=new TLegend(0.2,0.6,0.5,0.9);
 leg->SetBorderSize(0);
 leg->AddEntry(g_g4,"Geant4","lpe");
 leg->AddEntry(g_fast,"FastSim","lpe");
 leg->Draw();
 can->SetLogx();
 can->SetLogy();
 can->SetGridx();
 can->SetGridy();
 can->Print(Form("plots/energy_%s.pdf",type.c_str()));
 
 TCanvas* can2=new TCanvas("can2","can2",0,0,800,600);
 g_g4_ratio->Draw("alp");
 g_g4_ratio->GetXaxis()->SetTitle(Form("True energy [MeV] #bf{%s}",type.c_str()));
 g_g4_ratio->GetYaxis()->SetTitle("Simulated energy/true energy");
 g_fast_ratio->SetMarkerColor(2); g_fast_ratio->SetLineStyle(2); g_fast_ratio->SetLineColor(2); g_fast_ratio->SetMarkerStyle(4);
 g_fast_ratio->Draw("lpsame");
 TLegend* leg2=new TLegend(0.6,0.2,0.9,0.45);
 leg2->SetBorderSize(0);
 leg2->AddEntry(g_g4_ratio,"Geant4","lpe");
 leg2->AddEntry(g_fast_ratio,"FastSim","lpe");
 leg2->Draw();
 can2->SetLogx();
 //can2->SetLogy();
 can2->SetGridx();
 can2->SetGridy();
 can2->Print(Form("plots/energy_%s_ratio.pdf",type.c_str()));
 
}


void get_energy(int dsid, double &energy_val_g4, double &energy_val_fast, double &energy_rms_g4, double &energy_rms_fast)
{
 
 string version="ver01";

  TFile* file=TFile::Open(Form("output/ds%i.eparavali.%s.root",dsid,version.c_str()));
  
  //get layers:
  TFile* file1=TFile::Open(Form("output/ds%i.firstPCA.%s.root",dsid,version.c_str()));
  TH2I* h_layer=(TH2I*)file1->Get("h_layer");
  int pcabins=h_layer->GetNbinsX();
  vector<int> layerNr;
  for(int i=1;i<=h_layer->GetNbinsY();i++)
  {
 	 if(h_layer->GetBinContent(1,i)==1) 
 	  layerNr.push_back(h_layer->GetYaxis()->GetBinCenter(i));
  }
  vector<string> layer;
  for(unsigned int l=0;l<layerNr.size();l++)
   layer.push_back(Form("layer%i",layerNr[l]));
  layer.push_back("totalE");
  
  vector<string> name;
  vector<string> title;
  for(unsigned int l=0;l<layer.size()-1;l++)
  {
   cout<<"layer "<<layer[l]<<endl;
   name.push_back(layer[l].c_str());
   title.push_back(Form("Energy fraction in Layer %i",layerNr[l]));
  }
  name.push_back("Total energy");  title.push_back("total E [MeV]");
  
  
  for(unsigned int l=0;l<layer.size();l++)
  {
   cout<<"now do plots for layer "<<layer[l]<<endl;
  	
   double min,max,rmin,rmax;
   TH1D* h_output_lin;
   TH1D* h_input_lin;
   	h_output_lin=(TH1D*)file->Get(Form("h_output_%s",layer[l].c_str())); h_output_lin->SetName("h_output_lin");
   	h_input_lin =(TH1D*)file->Get(Form("h_input_%s",layer[l].c_str()));  h_input_lin->SetName("h_input_lin");
   
   if(l==layer.size()-1)
   {
    energy_val_g4  =h_input_lin->GetMean();
    energy_val_fast=h_output_lin->GetMean();
    energy_rms_g4  =h_input_lin->GetRMS();
    energy_rms_fast=h_output_lin->GetRMS();
   }
 } //for layers
 
 file1->Close();
 file->Close();
 
}

