/*
  Copyright (C) 2002-2017 CERN for the benefit of the ATLAS collaboration
*/

#include "CaloGeometryFromFile.h"
#include <TTree.h>
#include <TFile.h>
#include "CaloDetDescr/CaloDetDescrElement.h"
#include <fstream>
#include <sstream>
#include <TGraph.h>
#include "TH2D.h"

using namespace std;

map<unsigned long long, unsigned long long> g_cellId_vs_cellHashId_map;

CaloGeometryFromFile::CaloGeometryFromFile() : CaloGeometry()
{
}

CaloGeometryFromFile::~CaloGeometryFromFile()
{
}

bool CaloGeometryFromFile::LoadGeometryFromFile(TString filename,TString treename,TString hashfile)
{
	ifstream textfile(hashfile);
	unsigned long long id, hash_id; 
	cout << "Loading cellId_vs_cellHashId_map" << endl;
	int i=0;
	string line;
	stringstream s;
	while(1){
		//getline(textfile,line);
		s.str(line);
	  textfile >> id;
		if(textfile.eof())break;
		textfile >> hash_id;
		g_cellId_vs_cellHashId_map[id]=hash_id;
		if(i%10000==0)cout << "Line: " << i << " line " << line << " id " << hex << id << " hash_id " << dec << hash_id << endl;
		i++;
	}
	cout << "Done." << endl;
	

  TTree *tree;
  TFile *f = TFile::Open(filename);
  if(!f) return false;
  f->GetObject(treename,tree);
  if(!tree) return false;

  TTree* fChain = tree;
  
  CaloDetDescrElement cell;
  
  // List of branches
  TBranch        *b_identifier;   //!
  TBranch        *b_calosample;   //!
  TBranch        *b_eta;   //!
  TBranch        *b_phi;   //!
  TBranch        *b_r;   //!
  TBranch        *b_eta_raw;   //!
  TBranch        *b_phi_raw;   //!
  TBranch        *b_r_raw;   //!
  TBranch        *b_x;   //!
  TBranch        *b_y;   //!
  TBranch        *b_z;   //!
  TBranch        *b_x_raw;   //!
  TBranch        *b_y_raw;   //!
  TBranch        *b_z_raw;   //!
  TBranch        *b_deta;   //!
  TBranch        *b_dphi;   //!
  TBranch        *b_dr;   //!
  TBranch        *b_dx;   //!
  TBranch        *b_dy;   //!
  TBranch        *b_dz;   //!
  
  fChain->SetMakeClass(1);
  fChain->SetBranchAddress("identifier", &cell.m_identify, &b_identifier);
  fChain->SetBranchAddress("calosample", &cell.m_calosample, &b_calosample);
  fChain->SetBranchAddress("eta", &cell.m_eta, &b_eta);
  fChain->SetBranchAddress("phi", &cell.m_phi, &b_phi);
  fChain->SetBranchAddress("r", &cell.m_r, &b_r);
  fChain->SetBranchAddress("eta_raw", &cell.m_eta_raw, &b_eta_raw);
  fChain->SetBranchAddress("phi_raw", &cell.m_phi_raw, &b_phi_raw);
  fChain->SetBranchAddress("r_raw", &cell.m_r_raw, &b_r_raw);
  fChain->SetBranchAddress("x", &cell.m_x, &b_x);
  fChain->SetBranchAddress("y", &cell.m_y, &b_y);
  fChain->SetBranchAddress("z", &cell.m_z, &b_z);
  fChain->SetBranchAddress("x_raw", &cell.m_x_raw, &b_x_raw);
  fChain->SetBranchAddress("y_raw", &cell.m_y_raw, &b_y_raw);
  fChain->SetBranchAddress("z_raw", &cell.m_z_raw, &b_z_raw);
  fChain->SetBranchAddress("deta", &cell.m_deta, &b_deta);
  fChain->SetBranchAddress("dphi", &cell.m_dphi, &b_dphi);
  fChain->SetBranchAddress("dr", &cell.m_dr, &b_dr);
  fChain->SetBranchAddress("dx", &cell.m_dx, &b_dx);
  fChain->SetBranchAddress("dy", &cell.m_dy, &b_dy);
  fChain->SetBranchAddress("dz", &cell.m_dz, &b_dz);
  
  Long64_t nentries = fChain->GetEntriesFast();
  for (Long64_t jentry=0; jentry<nentries;jentry++) {
    Long64_t ientry = fChain->LoadTree(jentry);
    if (ientry < 0) break;
    fChain->GetEntry(jentry);
    
    if (g_cellId_vs_cellHashId_map.find(cell.m_identify)!=g_cellId_vs_cellHashId_map.end()) {
      cell.m_hash_id=g_cellId_vs_cellHashId_map[cell.m_identify];
      if(cell.m_hash_id!=jentry) cout<<jentry<<" : ERROR hash="<<cell.m_hash_id<<endl;
    }  
    else cout << endl << "ERROR: Cell id not found in the cellId_vs_cellHashId_map!!!" << endl << endl;
    

    const CaloDetDescrElement* pcell=new CaloDetDescrElement(cell);
    this->addcell(pcell);
		
    if(jentry%25000==0) {
    //if(jentry==nentries-1) {
			cout << "Checking loading cells from file" << endl;
      cout<<jentry<<" : "<<pcell->getSampling()<<", "<<pcell->identify()<<endl;

      
      
      
      //if(jentry>5) break;
    }
  }
	//cout<<"all : "<<m_cells.size()<<endl;
  //unsigned long long max(0);
  //unsigned long long min_id=m_cells_in_sampling[0].begin()->first;
  //for(int i=0;i<21;++i) {
		////cout<<"  cells sampling "<<i<<" : "<<m_cells_in_sampling[i].size()<<" cells";
		////cout<<", "<<m_cells_in_regions[i].size()<<" lookup map(s)"<<endl;
		//for(auto it=m_cells_in_sampling[i].begin(); it!=m_cells_in_sampling[i].end();it++){
			////cout << it->second->getSampling() << " " << it->first << endl;
			//if(min_id/10 >=  it->first){
				////cout << "Warning: Identifiers are not in increasing order!!!!" << endl;
				////cout << min_id << " " << it->first << endl;
				
			//}
			//if(min_id > it->first)min_id = it->first;
		//}
	//}
	//cout << "Min id for samplings < 21: " << min_id << endl;
  delete f;
	//return true;
	bool ok = PostProcessGeometry();
	
	cout << "Result of PostProcessGeometry(): " << ok << endl;

	const CaloDetDescrElement* mcell=0;
	unsigned long long cellid64(3179554531063103488);
	Identifier cellid(cellid64);
	mcell=this->getDDE(cellid); //This is working also for the FCal
	
	std::cout << "\n \n";
	std::cout << "Testing whether CaloGeoGeometry is loaded properly" << std::endl;
	if(!mcell)std::cout << "Cell is not found" << std::endl;
	std::cout << "Identifier " << mcell->identify() <<" sampling " << mcell->getSampling() << " eta: " << mcell->eta() << " phi: " << mcell->phi() << " CaloDetDescrElement="<<mcell << std::endl<< std::endl;
	
	const CaloDetDescrElement* mcell2 = this->getDDE(mcell->getSampling(),mcell->eta(),mcell->phi());
	std::cout << "Identifier " << mcell2->identify() <<" sampling " << mcell2->getSampling() << " eta: " << mcell2->eta() << " phi: " << mcell2->phi() << " CaloDetDescrElement="<<mcell2<< std::endl<< std::endl;
	
  return ok;
}

bool CaloGeometryFromFile::LoadFCalGeometryFromFiles(TString filename1,TString filename2,TString filename3){

  vector<ifstream*> electrodes(3);

  electrodes[0]=new ifstream(filename1);
  electrodes[1]=new ifstream(filename2);
  electrodes[2]=new ifstream(filename3);


  int    thisTubeId;
  int    thisTubeI;
  int    thisTubeJ;
  //int    thisTubeID;
  //int    thisTubeMod;
  double thisTubeX;
  double thisTubeY;
  TString tubeName;

  //int second_column;
  string seventh_column;
  string eight_column;
  int ninth_column;





  int i;
  for(int imodule=1;imodule<=3;imodule++){

    i=0;
    while(1){

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

      stringstream s;


      m_FCal_ChannelMap.add_tube(tubeNamestring, imodule, thisTubeId, thisTubeI,thisTubeJ, thisTubeX, thisTubeY,seventh_column);
      
      i++;
    }
  }


  m_FCal_ChannelMap.finish(); // Creates maps
  
  for(int imodule=1;imodule<=3;imodule++) delete electrodes[imodule-1];
  electrodes.clear();

  this->calculateFCalRminRmax();
  return this->checkFCalGeometryConsistency();

}

void CaloGeometryFromFile::DrawFCalGraph(int isam,int color){
	
	stringstream ss;
	ss << "FCal" << isam - 20 << endl;
	TString name = ss.str().c_str();
	
	const int size=m_cells_in_sampling[isam].size();
	
	std::vector<double> x;
	std::vector<double> y;
	x.reserve(size);
	y.reserve(size);

	//const CaloDetDescrElement* cell;

	for(auto it=m_cells_in_sampling[isam].begin();it!=m_cells_in_sampling[isam].end();it++){
		x.push_back(it->second->x());
		y.push_back(it->second->y());
	}
	// cout << size << endl;
	//TH2D* h = new TH2D("","",10,-0.5,0.5,10,-0.5,0.5);
	//h->SetStats(0);
	//h->Draw();
	TGraph* graph = new TGraph(size, &x[0], &y[0]);
	graph->SetLineColor(color);
	graph->SetTitle(name);
	graph->SetMarkerStyle(21);
	graph->SetMarkerColor(color);
	graph->SetMarkerSize(0.5);
	graph->GetXaxis()->SetTitle("x [mm]");
	graph->GetYaxis()->SetTitle("y [mm]");

	graph->Draw("AP");
}	
	
		
