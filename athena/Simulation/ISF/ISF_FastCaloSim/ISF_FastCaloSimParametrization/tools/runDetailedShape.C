/*
  Copyright (C) 2002-2017 CERN for the benefit of the ATLAS collaboration
*/

void runDetailedShape(){

  gSystem->AddIncludePath(" -I.. ");
  gROOT->LoadMacro("../src/CaloGeometry.cxx+");
  gROOT->LoadMacro("CaloGeometryFromFile.cxx+");
  gROOT->LoadMacro("../Root/DetailedShapeBinning.cxx+");
  gROOT->LoadMacro("../Root/TFCS2DFunction.cxx+");
  gROOT->LoadMacro("../Root/TFCS2DFunctionRegression.cxx+");
  gROOT->LoadMacro("../Root/TFCS2Function.cxx+");
  gROOT->LoadMacro("../Root/FitDetailedShape.cxx+");
  
  bool runBinning  = false;
  bool runNNfit    = true;

  bool runmode1    = true;
  bool runmode2    = false;
  bool ismatched   = false;  
  bool doNNHit     = false;

  string particle="pi";
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

  // type of input file you want to run on
  if(!ismatched) inputfile="/afs/cern.ch/work/c/conti/public/AF2/ISF_HitAnalysis_evgen_calo__"+PID+"_E50000_50000_eta20_25_Evts0-5500_vz_0_origin_calo.standard.pool.root";
    
  //string inputfile="root:://eos/atlas/user/c/conti/atlasreadable/AF2/ISF_HitAnalysis_evgen_calo__"+PID+"_E50000_50000_eta20_25_Evts0-5500_vz_0_origin_calo.standard.pool.root";

  if(ismatched){ 
  //inputfile="/afs/cern.ch/work/c/conti/private/ISF_FastCaloSimParametrization/INPUT/user.zmarshal.8071918._000001.matched_output.root"; 
    inputfile="root://eosatlas//eos/atlas/atlasgroupdisk/soft-simul/rucio/user/zmarshal/00/24/user.zmarshal.7814824._000001.matched_output.root"; //v1_w2_160301
    inputfile="root://eosatlas//eos/atlas/atlasgroupdisk/soft-simul/rucio/user/zmarshal/4d/19/user.zmarshal.7805067._000001.matched_output.root"; //v0_w2_160229
    inputfile="root://eosatlas//eos/atlas/atlasgroupdisk/soft-simul/rucio/user/zmarshal/78/74/user.zmarshal.8071918._000001.matched_output.root"; //w0_160406_5mmMerge
    inputfile="root://eosatlas//eos/atlas/atlasgroupdisk/soft-simul/rucio/user/zmarshal/44/b4/user.zmarshal.8071920._000001.matched_output.root"; //w0_160406_1mmMerge
  }


  float layer=2;
  float PCAbin=0;
  int nybinsR=50;
  float mincalosize=-1000;
  
  ostringstream os;
  os << layer ;
  ostringstream os2;
  os2 << PCAbin ;
  string labeloutput  = "50GeV_"+particle+"_layer"+os.str()+"_PCAbin"+os2.str();
  string labeltitle   = "50 GeV "+sparticle+"^{#pm}, layer "+os.str()+", bin(PCA)="+os2.str();
  
  float cellr=0;
  float cellz=0;
  float celleta=0;
  float cell_deta=0;
  float cell_dphi=0;

  // Determine the binning
  if(runBinning){
    DetailedShapeBinning* mydetailedshape=new DetailedShapeBinning();
    bool test = mydetailedshape->run(false,ismatched,inputfile,nybinsR,layer, particle, PCAbin, labeltitle, labeloutput,mincalosize,cell_deta,cell_dphi,cellr,cellz,celleta);
    // Test cell position
    std::cout << "CELL POSITION IS : " << cell_deta << " " << cell_dphi << " " << cellr << " " << cellz << " " << celleta << std::endl;
  }    

  // To perform the NN fit of the shower shape 
  if(runNNfit){
    FitDetailedShape* myfit = new FitDetailedShape(); 
    string outfile = "output/DetailedShape_"+labeloutput+".root";

    if(runmode1){
      //!doNNHit : Fit shower shape - LN(energy density) 
      // doNNHit : Fit shower shape - LN(energy) to generate many hits
      myfit->run(outfile,labeltitle,labeloutput,1,mincalosize,doNNHit);
      if(doNNHit){
      	DetailedShapeBinning* mydetailedshape2=new DetailedShapeBinning();
      	string inputfileNNHit = "output/NNHit_"+labeloutput+".root";
      	bool test2 = mydetailedshape2->run(true,ismatched,inputfileNNHit,nybinsR,layer, particle, PCAbin, labeltitle, labeloutput,mincalosize,cell_deta,cell_dphi,cellr,cellz,celleta);  
      }
    }
    // Run the second binning
    if(runmode2){
      if(doNNHit)
	outfile = "output/DetailedShape_"+labeloutput+"_NNHits.root";
      std::cout << "RUNNING WITH FILE : " << outfile << std::endl;
      myfit->run(outfile,labeltitle,labeloutput,2,mincalosize,doNNHit);
    }
  }

  std::cout << "END " <<  std::endl;

}

