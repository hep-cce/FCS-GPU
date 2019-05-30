/*
  Copyright (C) 2002-2017 CERN for the benefit of the ATLAS collaboration
*/

void drawCaloGeometry()
{
  // Warning: cell lookup in the FCal is not working yet!
  CaloGeometryFromFile* geo=new CaloGeometryFromFile();
  geo->SetDoGraphs(1);
  
  geo->LoadGeometryFromFile("/afs/cern.ch/atlas/groups/Simulation/FastCaloSimV2/Geometry-ATLAS-R2-2016-01-00-01.root", "ATLAS-R2-2016-01-00-01");
  TString path_to_fcal_geo_files = "/afs/cern.ch/atlas/groups/Simulation/FastCaloSimV2/";
  geo->LoadFCalGeometryFromFiles(path_to_fcal_geo_files + "FCal1-electrodes.sorted.HV.09Nov2007.dat", path_to_fcal_geo_files + "FCal2-electrodes.sorted.HV.April2011.dat", path_to_fcal_geo_files + "FCal3-electrodes.sorted.HV.09Nov2007.dat");

  //geo->Validate(10);
  //return;
  
  const CaloDetDescrElement* cell;
  cell=geo->getDDE(2,0.24,0.24); //This is not working yet for the FCal!!!
  cout<<"Found cell id="<<cell->identify()<<" sample="<<cell->getSampling()<<" eta="<<cell->eta()<<" phi="<<cell->phi()<<endl;
  
  Long64_t cellid64(3179554531063103488);
  Identifier cellid(cellid64);
  cell=geo->getDDE(cellid); //This is working also for the FCal
  cout<<"Found cell id="<<cell->identify()<<" sample="<<cell->getSampling()<<" eta="<<cell->eta()<<" phi="<<cell->phi()<<endl;
  
  new TCanvas("Calo_layout","Calo layout");
  TH2D* hcalolayout=new TH2D("hcalolayout","hcalolayout",50,-7000,7000,50,0,4000);
  hcalolayout->Draw();
  hcalolayout->SetStats(0);
  
  TLegend* leg=new TLegend(0.30,0.13,0.70,0.37);
  leg->SetFillStyle(0);
  leg->SetFillColor(10);
  leg->SetBorderSize(1);
  leg->SetNColumns(2);

  for(int i=0;i<24;++i) {
    double eta=0.2;
    double mineta,maxeta;
    geo->minmaxeta(i,eta,mineta,maxeta);
    cout<<geo->SamplingName(i)<<" : mineta="<<mineta<<" maxeta="<<maxeta<<endl;
    if(mineta<eta && maxeta>eta) {
      double avgeta=eta;
      cout<<"  r="<<geo->rent(i,avgeta)<<" -> "<<geo->rmid(i,avgeta)<<" -> "<<geo->rext(i,avgeta)<<endl;
    }  
    geo->GetGraph(i)->Draw("Psame");
    std::string sname=Form("Sampling %2d : ",i);
    sname+=geo->SamplingName(i);
    leg->AddEntry(geo->GetGraph(i),sname.c_str(),"LF");
  }
  leg->Draw();
  
  geo->DrawGeoForPhi0();
  
}

