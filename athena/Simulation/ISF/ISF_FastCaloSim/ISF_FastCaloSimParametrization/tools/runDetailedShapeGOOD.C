/*
  Copyright (C) 2002-2017 CERN for the benefit of the ATLAS collaboration
*/

#include <sstream>
#include <iostream>
#include <fstream>
#include <vector>
using namespace std;

void runDetailedShapeGOOD(){

  gSystem->AddIncludePath(" -I.. ");
  gROOT->LoadMacro("../src/CaloGeometry.cxx+");
  gROOT->LoadMacro("CaloGeometryFromFile.cxx+");
  gROOT->LoadMacro("../Root/DetailedShapeBinning.cxx+");
  gROOT->LoadMacro("../Root/TFCS2DFunction.cxx+");
  gROOT->LoadMacro("../Root/TFCS2DFunctionRegression.cxx+");
  gROOT->LoadMacro("../Root/TFCS2Function.cxx+");
  gROOT->LoadMacro("../Root/FitDetailedShape.cxx+");
  //
  string particle="e";
  string PID = "11";
  string sparticle = "e";
  if(particle=="pi"){
    sparticle="#pi";
    PID="211";
  }
  if(particle=="gamma"){
    sparticle="#gamma";
    PID="22";
  }
  
  string inputfile = "";
  
  
  bool doNNHit=0; // has to be false for the first iteration !!!

  std::vector<float> etaslice;

  // type of input file you want to run on
  bool ismatched = 1;
  
  if(!ismatched)
  {
    TChain* mychain=new TChain("ISF_HitAnalysis/CaloHitAna");
    if(PID=="211") mychain->Add(("/afs/cern.ch/work/c/conti/public/AF2/ISF_HitAnalysis_evgen_calo__"+PID+"_E50000_50000_eta20_25_Evts0-5500_vz_0_origin_calo.standard.pool.root").c_str());
    if(PID=="11")  mychain->Add(("/afs/cern.ch/work/c/conti/public/AF2/ISF_HitAnalysis_evgen_calo__"+PID+"_E50000_50000_eta20_25_Evts0-5500_vz_0_origin_calo.pool.root").c_str());
    //string inputfile="root:://eos/atlas/user/c/conti/atlasreadable/AF2/ISF_HitAnalysis_evgen_calo__"+PID+"_E50000_50000_eta20_25_Evts0-5500_vz_0_origin_calo.standard.pool.root";
  }

  if(ismatched)
  { 
    //inputfile="/afs/cern.ch/work/c/conti/private/ISF_FastCaloSimParametrization/INPUT/user.zmarshal.8071918._000001.matched_output.root"; 
    //inputfile="root://eosatlas//eos/atlas/atlasgroupdisk/soft-simul/rucio/user/zmarshal/00/24/user.zmarshal.7814824._000001.matched_output.root"; //v1_w2_160301
    //inputfile="root://eosatlas//eos/atlas/atlasgroupdisk/soft-simul/rucio/user/zmarshal/4d/19/user.zmarshal.7805067._000001.matched_output.root"; //v0_w2_160229
    // GOOD : inputfile="root://eosatlas//eos/atlas/atlasgroupdisk/soft-simul/rucio/user/zmarshal/78/74/user.zmarshal.8071918._000001.matched_output.root"; //w0_160406_5mmMerge
    //inputfile="root://eosatlas//eos/atlas/atlasgroupdisk/soft-simul/rucio/user/zmarshal/44/b4/user.zmarshal.8071920._000001.matched_output.root"; //w0_160406_1mmMerge

    // Dataset 1 - optimized merging scheme
    TChain* mychain=new TChain("FCS_ParametrizationInput");
    mychain->Add("root://eosatlas//eos/atlas/atlasgroupdisk/soft-simul/rucio/user/fladias/88/05/user.fladias.8834798._000001.matched_output.root");
    mychain->Add("root://eosatlas//eos/atlas/atlasgroupdisk/soft-simul/rucio/user/fladias/cb/ec/user.fladias.8834798._000002.matched_output.root");
    mychain->Add("root://eosatlas//eos/atlas/atlasgroupdisk/soft-simul/rucio/user/fladias/d6/d5/user.fladias.8834798._000003.matched_output.root");
    mychain->Add("root://eosatlas//eos/atlas/atlasgroupdisk/soft-simul/rucio/user/fladias/b0/a6/user.fladias.8834798._000004.matched_output.root");
    mychain->Add("root://eosatlas//eos/atlas/atlasgroupdisk/soft-simul/rucio/user/fladias/6f/4f/user.fladias.8834798._000005.matched_output.root");
    mychain->Add("root://eosatlas//eos/atlas/atlasgroupdisk/soft-simul/rucio/user/fladias/5d/6c/user.fladias.8834798._000006.matched_output.root");
    mychain->Add("root://eosatlas//eos/atlas/atlasgroupdisk/soft-simul/rucio/user/fladias/e8/71/user.fladias.8834798._000007.matched_output.root");
    mychain->Add("root://eosatlas//eos/atlas/atlasgroupdisk/soft-simul/rucio/user/fladias/15/9f/user.fladias.8834798._000008.matched_output.root");

    // Dataset 2 - 1mm merging scheme
    //mychain->Add("root://eosatlas//eos/atlas/atlasgroupdisk/soft-simul/rucio/user/fladias/76/8c/user.fladias.8834800._000001.matched_output.root");
    //mychain->Add("root://eosatlas//eos/atlas/atlasgroupdisk/soft-simul/rucio/user/fladias/25/5c/user.fladias.8834800._000002.matched_output.root");
    //mychain->Add("root://eosatlas//eos/atlas/atlasgroupdisk/soft-simul/rucio/user/fladias/7c/11/user.fladias.8834800._000003.matched_output.root");
    //mychain->Add("root://eosatlas//eos/atlas/atlasgroupdisk/soft-simul/rucio/user/fladias/4e/1f/user.fladias.8834800._000004.matched_output.root");
    //mychain->Add("root://eosatlas//eos/atlas/atlasgroupdisk/soft-simul/rucio/user/fladias/ca/6b/user.fladias.8834800._000005.matched_output.root");
    //mychain->Add("root://eosatlas//eos/atlas/atlasgroupdisk/soft-simul/rucio/user/fladias/68/51/user.fladias.8834800._000006.matched_output.root");
    //mychain->Add("root://eosatlas//eos/atlas/atlasgroupdisk/soft-simul/rucio/user/fladias/4d/ea/user.fladias.8834800._000007.matched_output.root");
    //mychain->Add("root://eosatlas//eos/atlas/atlasgroupdisk/soft-simul/rucio/user/fladias/ed/d9/user.fladias.8834800._000008.matched_output.root");
    //mychain->Add("root://eosatlas//eos/atlas/atlasgroupdisk/soft-simul/rucio/user/fladias/f7/0e/user.fladias.8834800._000009.matched_output.root");
    //mychain->Add("root://eosatlas//eos/atlas/atlasgroupdisk/soft-simul/rucio/user/fladias/6e/67/user.fladias.8834800._000010.matched_output.root");
    //mychain->Add("root://eosatlas//eos/atlas/atlasgroupdisk/soft-simul/rucio/user/fladias/4a/f5/user.fladias.8834800._000011.matched_output.root");
    //mychain->Add("root://eosatlas//eos/atlas/atlasgroupdisk/soft-simul/rucio/user/fladias/70/a8/user.fladias.8834800._000012.matched_output.root");
    //mychain->Add("root://eosatlas//eos/atlas/atlasgroupdisk/soft-simul/rucio/user/fladias/96/61/user.fladias.8834800._000013.matched_output.root");
    //mychain->Add("root://eosatlas//eos/atlas/atlasgroupdisk/soft-simul/rucio/user/fladias/d6/f0/user.fladias.8834800._000014.matched_output.root");
    //mychain->Add("root://eosatlas//eos/atlas/atlasgroupdisk/soft-simul/rucio/user/fladias/8d/eb/user.fladias.8834800._000015.matched_output.root");
    //mychain->Add("root://eosatlas//eos/atlas/atlasgroupdisk/soft-simul/rucio/user/fladias/60/f3/user.fladias.8834800._000016.matched_output.root");
    //mychain->Add("root://eosatlas//eos/atlas/atlasgroupdisk/soft-simul/rucio/user/fladias/07/71/user.fladias.8834800._000017.matched_output.root");
    //mychain->Add("root://eosatlas//eos/atlas/atlasgroupdisk/soft-simul/rucio/user/fladias/a1/b6/user.fladias.8834800._000018.matched_output.root");
    //mychain->Add("root://eosatlas//eos/atlas/atlasgroupdisk/soft-simul/rucio/user/fladias/12/2b/user.fladias.8834800._000019.matched_output.root");
    //mychain->Add("root://eosatlas//eos/atlas/atlasgroupdisk/soft-simul/rucio/user/fladias/0b/bc/user.fladias.8834800._000020.matched_output.root");
    //mychain->Add("root://eosatlas//eos/atlas/atlasgroupdisk/soft-simul/rucio/user/fladias/13/43/user.fladias.8834800._000021.matched_output.root");

    // whih eta slice to consider 
    etaslice.push_back(0.20);
    etaslice.push_back(0.25);
  }

  float layer=2;
  int PCAbin=1;
  int nybinsR=20;
  float mincalosize=-1000;
  
  ostringstream os;
  os << layer ;
  ostringstream os2;
  os2 << PCAbin ;
  string labeloutput  = "50GeV_"+particle+"_layer"+os.str()+"_PCAbin"+os2.str();
  string labeltitle   = "50 GeV "+sparticle+"^{#pm}, layer "+os.str()+", bin(PCA)="+os2.str();
  
  // geometry
  float cell_deta=0;
  float cell_dphi=0;
  float cellr=0;
  float cellz=0;
  float celleta=0;

  //// Determine the binning
  DetailedShapeBinning* mydetailedshape=new DetailedShapeBinning();
  bool test = mydetailedshape->run(doNNHit,ismatched,mychain,nybinsR,layer, particle, PCAbin, labeltitle, labeloutput,mincalosize,cell_deta,cell_dphi,cellr,cellz,celleta,etaslice);
  //
  ////// To perform the NN fit of the shower shape 
  float calosize = 15;
  //string outfile = "output/DetailedShape_"+labeloutput+"_NNHits.root";
  string outfile = "output/DetailedShape_"+labeloutput+".root";
  FitDetailedShape* myfit = new FitDetailedShape(); 
  int mycase = 0 ; 
  ////
  //// Fit shower shape 
  mycase = 1 ; 
  int verbose_level = 1;
  myfit->run(outfile,verbose_level,labeltitle,labeloutput,mycase,mincalosize,doNNHit);
  //// To create the file based on the sim hits
  //doNNHit = 1 ; 
  //myfit->run(outfile,verbose_level,labeltitle,labeloutput,mycase,mincalosize,doNNHit);
  //////// Fit input variables 
  //mycase = 2 ; 
  //doNNHit = 0 ; 
  //myfit->run(outfile,verbose_level,labeltitle,labeloutput,mycase,mincalosize,doNNHit);
  ////

  // Redo analysis with simulated hits for mycase=2
  //
  //doNNHit = 1 ;
  //inputfile = "/afs/cern.ch/work/c/conti/private/ISF_FastCaloSimParametrization/tools/output/NNHit_50GeV_pi_layer2_PCAbin0.root";
  //bool test = mydetailedshape->run(doNNHit,ismatched,inputfile,nybinsR,layer, particle, PCAbin, labeltitle, labeloutput,mincalosize,cell_deta,cell_dphi,cellr,cellz,celleta);
  //
  //myfit->run(outfile,verbose_level,labeltitle,labeloutput,mycase,mincalosize,doNNHit);
  //
  std::cout << "END " <<  std::endl;

}

