/*
  Copyright (C) 2002-2021 CERN for the benefit of the ATLAS collaboration
*/

void check_energy_high()
{
 
 int pdg=22;
 
 vector<string> dsid; vector<string> name;
 vector<string> hname;
 if(pdg==22)
 {
  dsid.push_back("431404"); name.push_back("#gamma 1 TeV");
  dsid.push_back("431504"); name.push_back("#gamma 2 TeV");
  dsid.push_back("431604"); name.push_back("#gamma 4 TeV");
  hname.push_back("layer1");
  hname.push_back("layer2");
  hname.push_back("layer3");
  hname.push_back("layer12");
 }
 if(pdg==211)
 {
  dsid.push_back("434804"); name.push_back("#pi 1 TeV");
  dsid.push_back("434904"); name.push_back("#pi 2 TeV");
  dsid.push_back("435004"); name.push_back("#pi 4 TeV");
  hname.push_back("layer1");
  hname.push_back("layer2");
  hname.push_back("layer3");
  hname.push_back("layer12");
  hname.push_back("layer13");
  hname.push_back("layer14");
 }
 hname.push_back("totalEratio");
 
 for(unsigned int l=0;l<hname.size();l++)
 {
  TCanvas* can=new TCanvas("can","can",0,0,1200,600);
  can->Divide(2);
  TLegend* leg=new TLegend(0.5,0.95-0.05*dsid.size(),0.75,0.95);
  leg->SetFillStyle(0);
  leg->SetBorderSize(0);
  double ymax=0;
  for(unsigned int f=0;f<dsid.size();f++)
  {
   TFile* file=TFile::Open(Form("output/ds%s.eparavali.ver01.root",dsid[f].c_str()));
   TH1D* hist=(TH1D*)file->Get(Form("h_input_%s",hname[l].c_str())); hist->SetName("hist");
   hist->Rebin(5);
   hist->Scale(1.0/hist->Integral());
   if(hist->GetBinContent(hist->GetMaximumBin())>ymax)
    ymax=hist->GetBinContent(hist->GetMaximumBin());
  }
  for(unsigned int f=0;f<dsid.size();f++)
  {
   cout<<"histo "<<l<<" dsid "<<dsid[f]<<endl;
   TFile* file=TFile::Open(Form("output/ds%s.eparavali.ver01.root",dsid[f].c_str()));
   TH1D* hist=(TH1D*)file->Get(Form("h_input_%s",hname[l].c_str())); hist->SetName("hist");
   if(f==0) { hist->SetLineColor(1); hist->SetLineStyle(1); hist->SetMarkerColor(1); hist->SetMarkerStyle(8); }
   if(f==1) { hist->SetLineColor(2); hist->SetLineStyle(2); hist->SetMarkerColor(2); hist->SetMarkerStyle(4); }
   if(f==2) { hist->SetLineColor(7); hist->SetLineStyle(3); hist->SetMarkerColor(7); hist->SetMarkerStyle(30); }
   hist->Rebin(5);
   double sf=1.0/hist->Integral();
   hist->Scale(sf);
   for(int b=1;b<=hist->GetNbinsX();b++)
    hist->SetBinError(b,hist->GetBinError(b)*sf);
   TH1D* hist_log=(TH1D*)hist->Clone("hist_log");
   can->cd(1);
   if(f==0)
   {
    hist->Draw("e");
    hist->GetYaxis()->SetRangeUser(0.0001,ymax*1.2);
    hist->GetXaxis()->SetTitle(Form("%s",hname[l].c_str()));
    if(hname[l]=="totalEratio") hist->GetXaxis()->SetTitle("total E / truth E");
   }
   else
    hist->Draw("esame");
   can->cd(2);
   if(f==0)
   {
    hist_log->Draw("e");
    hist_log->GetYaxis()->SetRangeUser(0.0001,ymax*3.0);
    hist_log->GetXaxis()->SetTitle(Form("%s",hname[l].c_str()));
    if(hname[l]=="totalEratio") hist_log->GetXaxis()->SetTitle("total E / truth E");
   }
   else
    hist_log->Draw("esame");
   can->cd(2)->SetLogy();
   //file->Close();
   leg->AddEntry(hist,name[f].c_str(),"lpe");
  }
  leg->Draw();
  if(l==0)
   can->Print(Form("plots/highE/pdg%i.pdf(",pdg));
  if(l==hname.size()-1)
   can->Print(Form("plots/highE/pdg%i.pdf)",pdg));
  if(l>0 && l<hname.size()-1)
   can->Print(Form("plots/highE/pdg%i.pdf",pdg));
  can->Close();
  delete can;
 }
 
}

