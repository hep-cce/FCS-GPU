/*
  Copyright (C) 2002-2017 CERN for the benefit of the ATLAS collaboration
*/

#include "TChain.h"
#include <iostream>
#include "TGraph.h"

void run_geo();

void WriteGeneralInfo(TString /*cut_label*/, TString lumi, float size, float x, float y);
void ATLASLabel(Double_t x,Double_t y,const char* text, float tsize, Color_t color=1);
void WriteInfo(TString info, float size, float x, float y, int color=1);
void plotFCalCell(CaloGeometryFromFile* geo,int sampling, double x, double y);

void run_geo()
{
  
 //how to run in root 6 on lxplus:
 //.x init_geo.C+
 //.x run_geo.C
 
 // Warning: cell lookup in the FCal is not working yet!
 CaloGeometryFromFile* geo=new CaloGeometryFromFile();
 geo->SetDoGraphs(1);
 geo->LoadGeometryFromFile("/afs/cern.ch/atlas/groups/Simulation/FastCaloSimV2/Geometry-ATLAS-R2-2016-01-00-01.root", "ATLAS-R2-2016-01-00-01");
 TString path_to_fcal_geo_files = "/afs/cern.ch/atlas/groups/Simulation/FastCaloSimV2/";
 geo->LoadFCalGeometryFromFiles(path_to_fcal_geo_files + "FCal1-electrodes.sorted.HV.09Nov2007.dat", path_to_fcal_geo_files + "FCal2-electrodes.sorted.HV.April2011.dat", path_to_fcal_geo_files + "FCal3-electrodes.sorted.HV.09Nov2007.dat");
  //CaloGeometry::m_debug_identity=3179554531063103488;
	//geo->Validate();
 
 
  const CaloDetDescrElement* cell;
  cell=geo->getDDE(2,0.24,0.24); //This is not working yet for the FCal!!!
  //cout<<"Found cell id="<<cell->identify()<<" sample="<<cell->getSampling()<<" eta="<<cell->eta()<<" phi="<<cell->phi()<<endl;
  
  unsigned long long cellid64(3746994889972252672);
  Identifier cellid(cellid64);
  cell=geo->getDDE(cellid); //This is working also for the FCal
  
  cout<<"Found cell id="<<cell->identify()<<" sample="<<cell->getSampling()<<" eta="<<cell->eta()<<" phi="<<cell->phi()<<endl;
  
  /*TCanvas* canvas = new TCanvas("Calo_layout","Calo layout");
  TH2D* hcalolayout=new TH2D("hcalolayout","hcalolayout",50,-7000,7000,50,0,4000);
  hcalolayout->Draw();
  hcalolayout->SetStats(0);
  
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
  }
  
  geo->DrawGeoForPhi0();
  canvas->SaveAs("Calorimeter.png");*/
  
  float xInfo = 0.18;
  float  yInfo = 0.9;
  float sizeInfo=0.035;
    
	
  
  TCanvas* canvas = new TCanvas("FCal1_xy","FCal1_xy",600,600);
  geo->DrawFCalGraph(21,1);
  WriteGeneralInfo("","",sizeInfo,xInfo,yInfo);
  WriteInfo("FCal1", 0.05, 0.20, 0.21);
  canvas->SaveAs("FCal1Geometry.png");
  TCanvas* canvas2 = new TCanvas("FCal2_xy","FCal2_xy",600,600);
  geo->DrawFCalGraph(22,1);
  WriteGeneralInfo("","",sizeInfo,xInfo,yInfo);
  WriteInfo("FCal2", 0.05, 0.20, 0.21);
  canvas2->SaveAs("FCal2Geometry.png");
  TCanvas* canvas3 = new TCanvas("FCal3_xy","FCal3_xy",600,600);
  geo->DrawFCalGraph(23,1);
  WriteGeneralInfo("","",sizeInfo,xInfo,yInfo);
  WriteInfo("FCal3", 0.05, 0.20, 0.21);
  canvas3->SaveAs("FCal3Geometry.png");
  
  
  
  
  vector<ifstream*> electrodes(3);
  
  electrodes[0]=new ifstream((path_to_fcal_geo_files +"FCal1-electrodes.sorted.HV.09Nov2007.dat").Data());
  electrodes[1]=new ifstream((path_to_fcal_geo_files +"FCal2-electrodes.sorted.HV.April2011.dat").Data());
  electrodes[2]=new ifstream((path_to_fcal_geo_files +"FCal3-electrodes.sorted.HV.09Nov2007.dat").Data());
  
  
  
  int	thisTubeId;
  int    thisTubeI;
	int    thisTubeJ;
	//int    thisTubeID;
	int    thisTubeMod;
	double thisTubeX;
	double thisTubeY;
	TString tubeName;
	
	int second_column;
	string seventh_column;
	string eight_column;
	int ninth_column;
	
	FCAL_ChannelMap *cmap = new FCAL_ChannelMap(0);
	
	
	
	
	int i;
	for(int imodule=1;imodule<=3;imodule++){
		
		i=0;
		//while(i<50){
		while(1){
		
		  //cout << electrodes[imodule-1]->eof() << endl;
			(*electrodes[imodule-1]) >> tubeName;
			if(electrodes[imodule-1]->eof())break;
			(*electrodes[imodule-1]) >> thisTubeId; // ?????
			(*electrodes[imodule-1]) >> thisTubeI;
			(*electrodes[imodule-1]) >> thisTubeJ;
			(*electrodes[imodule-1]) >> thisTubeX;
			(*electrodes[imodule-1]) >> thisTubeY;
			(*electrodes[imodule-1]) >> seventh_column;
			(*electrodes[imodule-1]) >> eight_column;
			(*electrodes[imodule-1]) >> ninth_column;
			
			tubeName.ReplaceAll("'","");
			string tubeNamestring=tubeName.Data();
			
			std::istringstream tileStream1(std::string(tubeNamestring,1,1));
		  std::istringstream tileStream2(std::string(tubeNamestring,3,2));
		  std::istringstream tileStream3(std::string(tubeNamestring,6,3));
		  int a1=0,a2=0,a3=0;
		  if (tileStream1) tileStream1 >> a1;
		  if (tileStream2) tileStream2 >> a2;
		  if (tileStream3) tileStream3 >> a3;
			
			unsigned int tileName= (a3 << 16) + a2;
			stringstream s;
			
			
			cmap->add_tube(tubeNamestring, imodule, thisTubeId, thisTubeI,thisTubeJ, thisTubeX, thisTubeY,seventh_column);
			
			
			
			//cout << "FCal electrodes: " << tubeName << " " << second_column << " " << thisTubeI << " " << thisTubeJ << " " << thisTubeX << " " << thisTubeY << " " << seventh_column << " " << eight_column << " " << ninth_column << endl;
			//cout << tileStream1.str() << " " << tileStream2.str() << " " << tileStream3.str() << endl;
			//cout << a1 << " " << a2 << " " << a3 << " " << tileName << endl;
			i++;
		}
	}
	cmap->finish(); // Creates maps
	 
	//cmap->print_tubemap(1);
  //cmap->print_tubemap(2);
  //cmap->print_tubemap(3);
  
  int eta_index,phi_index;
  Long64_t eta_index64,phi_index64;
  //double x=423.755;
  //double y=123.41;
  double x=431.821;
  double y=116.694;
  //double x=21;
  //double y=27;
  
  
  //double x=436.892; 
  //double y=28.0237;
  
  
  const CaloDetDescrElement* mcell=0;
  const CaloDetDescrElement* mcell2=0;
  
  cout << endl;
  cout << "Looking for tile corresponding to [x,y] = [" <<  x << "," << y << "]" << endl;
  
  float* distance = new float(-99.);
  int* steps = new int(-99);
  
  for(int imap=1;imap<=3;imap++){
  
		cout << "Looking for tile in module " << imap << endl;
	  if(!cmap->getTileID(imap,x,y,eta_index,phi_index)){
			cout << "Not found" << endl;
	  }
	  else{
	    cout << "Tile found" << endl;
	    cout << "Tile Id " << (eta_index << 16) + phi_index << endl;
	    cout << "eta index: " << eta_index << endl;
	    cout << "phi index: " << phi_index << endl;
	    float dx,dy;
	    cmap->tileSize(imap, eta_index, phi_index, dx, dy);
	    cout << "Tile position: [x,y] = [" <<  cmap->x(imap,eta_index,phi_index) << "," << cmap->y(imap,eta_index,phi_index) << "]" << " [dx,dy] = [" << dx << "," << dy << "] " << endl;
	  }
	  mcell=geo->getFCalDDE(imap+20,x,y,+1,distance,steps);
	  cout << "Original hit position: [x,y] = [" <<  x << "," << y << "]" << endl;
	  cout << "Tile position from CaloGeometry: [x,y] = [" <<  mcell->x() << "," << mcell->y() << "]" << " [dx,dy] = [" << mcell->dx() << "," << mcell->dy() << "] " << " Identifier: " << mcell->identify() << endl;
	  cout << "Distance: " << *distance << endl;
	  cout << "Steps: " << *steps << endl;
		
	  Identifier identifier= mcell->identify();
	  
	
		/*eta_index64=eta_index;
		phi_index64=phi_index;
		if (imap==2) eta_index64+=100;
		if (imap==3) eta_index64+=200;
		cout << identifier << " " << (eta_index64 << 16) + phi_index64 << endl;
		
		
		
		mcell2=geo->getDDE((eta_index64 << 16) + phi_index64);
		cout << "Tile position for calogeometry using identifier: [x,y] = [" <<  mcell2->x() << "," << mcell2->y() << "]" << endl;*/
	
	
	
	}

  delete distance;
  delete steps;
  
  bool makeFCalCellPlots=false;

  if(makeFCalCellPlots){

    gSystem->mkdir("FCalCellShapes");
    
    for(int sampling=21;sampling<=23;sampling++){
      TString samplingSTR=Form("sampling_%i",sampling);
      gSystem->mkdir("FCalCellShapes/"+samplingSTR);
      for(auto it=cmap->begin(sampling-20);it!=cmap->end(sampling-20);it++){
      
	plotFCalCell(geo,sampling, it->second.x(),it->second.y());
      
      
      }
    }
  }
 
}

void plotFCalCell(CaloGeometryFromFile* geo,int sampling, double x_orig, double y_orig){
  const CaloDetDescrElement* mcell=0;
  mcell=geo->getFCalDDE(sampling,x_orig,y_orig,+1);
  if(!mcell){
    cout << "Error in plotFCalCell: Cell not found" << endl;
    return;
  }
  
  double x=mcell->x();
  double y=mcell->y();
  double dx=mcell->dx();
  double dy=mcell->dy();
  
  int N=1000;
  
  
  double yy= y+1.5*dy;
  double xx=x-dx;
  vector<double> vecx,vecy;
  
  for(int i=0;i<N+1;i++){
   
    
    
    for (int j=0;j<N+1;j++){
      xx+=(2.*dx)/N;
      mcell=geo->getFCalDDE(sampling,xx,yy,+1);
      if(!mcell || !TMath::AreEqualRel(mcell->x(),x,1.E-6) || !TMath::AreEqualRel(mcell->y(),y,1.E-6)){
	 //cout << " ";
      }
      else {
	//cout << "X";
	vecx.push_back(xx-x);
	vecy.push_back(yy-y);
      }
    }
    //cout << endl;
    
    yy -= 1.5*(2.*dy)/N;
    xx = x-dx;
    
  }
  
  double xmax= *std::max_element(vecx.begin(),vecx.end());
  double xmin= *std::min_element(vecx.begin(),vecx.end());
  double ymax= *std::max_element(vecy.begin(),vecy.end());
  double ymin= *std::min_element(vecy.begin(),vecy.end());
  
  xmax=std::max(xmax,dx);
  xmin=std::min(xmin,-dx);
  ymax=std::max(ymax,dy);
  ymin=std::min(ymin,-dy);
  
  TGraph* gr = new TGraph (vecx.size(),&vecx[0],&vecy[0]);
 
  TH2D* hdummy = new TH2D("h","",10,xmin,xmax,10,ymin,ymax);
 
  TCanvas* c = new TCanvas("Cell_shape","Cell_shape",600,600);
  c->cd();
  gr->SetLineColor(1);
  //gr->SetTitle(name);
  gr->SetMarkerStyle(21);
  gr->SetMarkerColor(1);
  gr->SetMarkerSize(0.5);
  //gr->GetXaxis()->SetRangeUser(-dx,dx);
  //gr->GetYaxis()->SetRangeUser(ymin,ymax);
  hdummy->GetXaxis()->SetTitle("#deltax [mm]");
  hdummy->GetYaxis()->SetTitle("#deltay [mm]");
  
  hdummy->Draw();
  gr->Draw("P");
  
  
  TString samplingSTR=Form("sampling_%i",sampling);
  TString s="FCalCellShapes/" + samplingSTR + "/FCalCellShape";
  s+=Form("_sampling%i_x%4.2f_y%4.2f",sampling,x,y);
  
  cout << s << endl;
  
  c->SaveAs(s+".png");
  
  delete hdummy;
  delete gr;
  delete c;
  
  
  
}



void ATLASLabel(Double_t x,Double_t y,const char* text, float tsize, Color_t color){
  TLatex l; //l.SetTextAlign(12);
  if (tsize>0) l.SetTextSize(tsize); 
  l.SetNDC();
  l.SetTextFont(72);
  l.SetTextColor(color);

  //double delx = 0.115*696*gPad->GetWh()/(472*gPad->GetWw());
  double delx = 0.14;

  l.DrawLatex(x,y,"ATLAS");
  if (text) {
    TLatex p; 
    p.SetNDC();
    if (tsize>0) p.SetTextSize(tsize); 
    p.SetTextFont(42);
    p.SetTextColor(color);
    p.DrawLatex(x+delx,y,text);
    //    p.DrawLatex(x,y,"#sqrt{s}=900GeV");
  }
}
void WriteGeneralInfo(TString /*cut_label*/, TString lumi, float size, float x, float y){
  TString label="";
  if (lumi=="") label+="  Simulation";
  //label+="  Internal";
    label+=" Preliminary";
  ATLASLabel(x,y,label.Data(),size*1.15);
  TString ToWrite="";
  TLatex l;
  l.SetNDC();
  l.SetTextFont(42);
  l.SetTextSize(size*0.9); 
  l.SetTextColor(1);
  //l.DrawLatex(x-0.005,y-0.07,cut_label.Data());

  double shift=0.55;
  //ToWrite="#sqrt{s}=13 TeV";
  //if (lumi!=""){
    ////ToWrite="L_{int}=";
    //ToWrite+=", ";
    //ToWrite+=lumi;
    //ToWrite+=" fb^{-1}";
    //shift=0.43;
  //}
  l.DrawLatex(x+shift,y,ToWrite.Data());
  
}
void WriteInfo(TString info, float size, float x, float y, int color){
  TLatex l;
  l.SetNDC();
  l.SetTextFont(42);
  l.SetTextSize(size); 
  l.SetTextColor(color);
  l.DrawLatex(x,y,info.Data());
}
