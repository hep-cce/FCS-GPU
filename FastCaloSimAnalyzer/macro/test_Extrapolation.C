void test_Extrapolation()
{
  CaloGeometryFromFile* geo=new CaloGeometryFromFile();
  geo->SetDoGraphs(1);
  
  geo->LoadGeometryFromFile("/afs/cern.ch/atlas/groups/Simulation/FastCaloSimV2/Geometry-ATLAS-R2-2016-01-00-01.root", "ATLAS-R2-2016-01-00-01");
  TString path_to_fcal_geo_files = "/afs/cern.ch/atlas/groups/Simulation/FastCaloSimV2/";
  geo->LoadFCalGeometryFromFiles(path_to_fcal_geo_files + "FCal1-electrodes.sorted.HV.09Nov2007.dat", path_to_fcal_geo_files + "FCal2-electrodes.sorted.HV.April2011.dat", path_to_fcal_geo_files + "FCal3-electrodes.sorted.HV.09Nov2007.dat");

  TFile* infile=TFile::Open("mc16_13TeV.photon.E1024.eta500_500.calohit.root");
  TTree* FCS_ParametrizationInput=(TTree*)infile->Get("FCS_ParametrizationInput");

  TCanvas* c;

  c=new TCanvas("IDCaloBoundary_rz","IDCaloBoundary: rz");
  FCS_ParametrizationInput->Draw("newTTC_IDCaloBoundary_r:newTTC_IDCaloBoundary_z>>gr_IDCaloBoundary_rz(1000,-4800,4800,1000,0,1200)");
  c->SaveAs(".png");
  
  for(int layer=-23;layer<=23;++layer) {
    TH2* htemp;
    int last=-1;
    int addfornext=1;
    TString title=Form("           %s (%d) : rz;z [mm];r [mm]",geo->SamplingName(abs(layer)).c_str(),abs(layer));
    TString title_pos=Form("           %s (%d) pos. z: rz;z [mm];r [mm]",geo->SamplingName(abs(layer)).c_str(),abs(layer));
    TString title_neg=Form("           %s (%d) neg. z: rz;z [mm];r [mm]",geo->SamplingName(abs(layer)).c_str(),abs(layer));
    
    if(layer>=0 && layer<=3) {
      htemp=new TH2F(Form("temp%d",layer),title,1000,-4000,4000,1000,1350,2000);
      last=3;
    }  
    if(layer>=-3 && layer<0) {
      continue;
    }  

    if(layer>=4 && layer<=7) {
      htemp=new TH2F(Form("temppos%d",layer),title_pos,1000,3600,4300,1000,0,2500);
      last=7;
    }  
    if(layer>=-7 && layer<=-4) {
      htemp=new TH2F(Form("tempneg%d",layer),title_neg,1000,-4300,-3600,1000,0,2500);
      last=7;
      addfornext=-1;
    }  

    if(layer>=8 && layer<=11) {
      htemp=new TH2F(Form("temppos%d",layer),title_pos,1000,4200,6400,1000,0,2500);
      last=11;
    }  
    if(layer>=-11 && layer<=-8) {
      htemp=new TH2F(Form("temppos%d",layer),title_neg,1000,-6400,-4200,1000,0,2500);
      last=11;
      addfornext=-1;
    }  

    if(layer>=12 && layer<=14) {
      htemp=new TH2F(Form("temp%d",layer),title,1000,-4000,4000,1000,2200,4000);
      last=3;
    }  
    if(layer>=-14 && layer<=-12) {
      continue;
    }  

    if(layer>=15 && layer<=16) {
      htemp=new TH2F(Form("temppos%d",layer),title_pos,1000,3000,4100,1000,2500,4000);
      last=16;
    }  
    if(layer==17) {
      htemp=new TH2F(Form("temppos%d",layer),title_pos,1000,3500,3600,1000,1000,3500);
    }  
    if(layer>=-16 && layer<=-15) {
      htemp=new TH2F(Form("temppos%d",layer),title_neg,1000,-4100,-3000,1000,2500,4000);
      last=16;
      addfornext=-1;
    }  
    if(layer==-17) {
      htemp=new TH2F(Form("temppos%d",layer),title_neg,1000,-3600,-3500,1000,1000,3500);
    }  

    if(layer>=18 && layer<=20) {
      htemp=new TH2F(Form("temppos%d",layer),title_pos,1000,3000,7000,1000,2000,4000);
      last=20;
    }  
    if(layer>=-20 && layer<=-18) {
      htemp=new TH2F(Form("temppos%d",layer),title_pos,1000,-7000,-3000,1000,2000,4000);
      last=20;
      addfornext=-1;
    }  

    if(layer>=21 && layer<=23) {
      htemp=new TH2F(Form("temppos%d",layer),title_pos,1000,4500,6500,1000,0,500);
      last=23;
    }  
    if(layer>=-23 && layer<=-21) {
      htemp=new TH2F(Form("temppos%d",layer),title_pos,1000,-6500,-4500,1000,0,500);
      last=23;
      addfornext=-1;
    }  

    if(layer>=0) c=new TCanvas(Form("layer%d_rzpos",abs(layer)),Form("layer %d : rz pos",abs(layer)));
     else  c=new TCanvas(Form("layer%d_rzneg",abs(layer)),Form("layer %d : rz neg",abs(layer)));
    htemp->SetStats(false);
    htemp->Draw();

    geo->DrawGeoSampleForPhi0(abs(layer),kGray,false);
    TGraph* gr_cells = (TGraph*)gPad->GetPrimitive("Graph");
    while(gr_cells) {
      gr_cells->SetName(Form("gr_cells%d",abs(layer)));
      gr_cells = (TGraph*)gPad->GetPrimitive("Graph");
    }  
    
    TLegend* leg=new TLegend(0.1,0.9,0.35,0.99,"","NDC");
    leg->SetNColumns(2);
    
    FCS_ParametrizationInput->Draw(Form("newTTC_entrance_r[0][%d]:newTTC_entrance_z[0][%d]",abs(layer),abs(layer)),"","same");
    TGraph* gr_ent = (TGraph*)gPad->GetPrimitive("Graph");
    gr_ent->SetName(Form("gr_layer%d_ent_rz",layer));
    gr_ent->SetLineColor(2);
    gr_ent->SetMarkerColor(2);
    gr_ent->SetMarkerStyle(7);
    gr_ent->SetMarkerSize(0.7);
    cout<<"gr_ent="<<gr_ent<<endl;
    leg->AddEntry(gr_ent,"Entrance","lp");

    FCS_ParametrizationInput->Draw(Form("newTTC_mid_r[0][%d]:newTTC_mid_z[0][%d]",abs(layer),abs(layer)),"","same");
    TGraph* gr_mid = (TGraph*)gPad->GetPrimitive("Graph");
    gr_mid->SetName(Form("gr_layer%d_mid_rz",layer));
    gr_mid->SetLineColor(1);
    gr_mid->SetMarkerStyle(7);
    gr_mid->SetMarkerSize(0.7);
    cout<<"gr_mid="<<gr_mid<<endl;
    leg->AddEntry(gr_mid,"Middle","lp");

    TGraph* gr_next = 0;
    if(abs(layer+addfornext)<=abs(last)) {
      FCS_ParametrizationInput->Draw(Form("newTTC_entrance_r[0][%d]:newTTC_entrance_z[0][%d]",abs(layer+addfornext),abs(layer+addfornext)),"","same");
      gr_next = (TGraph*)gPad->GetPrimitive("Graph");
      gr_next->SetName(Form("gr_layer%d_ent_rz",layer+addfornext));
      gr_next->SetLineColor(6);
      gr_next->SetMarkerColor(6);
      gr_next->SetMarkerStyle(4);
      gr_next->SetMarkerSize(0.4);
      cout<<"gr_next="<<gr_next<<endl;
    }
    
    FCS_ParametrizationInput->Draw(Form("newTTC_back_r[0][%d]:newTTC_back_z[0][%d]",abs(layer),abs(layer)),"","same");
    TGraph* gr_back = (TGraph*)gPad->GetPrimitive("Graph");
    gr_back->SetName(Form("gr_layer%d_back_rz",layer));
    gr_back->SetLineColor(kGreen+2);
    gr_back->SetMarkerColor(kGreen+2);
    gr_back->SetMarkerStyle(7);
    gr_back->SetMarkerSize(0.7);
    cout<<"gr_back="<<gr_back<<endl;
    leg->AddEntry(gr_back,"Back","lp");

    if(gr_next) leg->AddEntry(gr_next,"Ent. Next","p");
    
    leg->Draw();
    c->SaveAs(".png");
  }
}
