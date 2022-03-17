#include <fstream>
#include <iostream>
#include <sstream>

#include <docopt/docopt.h>

#include "CLHEP/Random/TRandomEngine.h"
#include "CLHEP/Random/RandFlat.h"

#include "ISF_FastCaloSimEvent/TFCSPCAEnergyParametrization.h"
#include "ISF_FastCaloSimEvent/TFCSExtrapolationState.h"
#include "ISF_FastCaloSimEvent/TFCSTruthState.h"

#include "secondPCA.h"
#include "TFCSApplyFirstPCA.h"
#include "TFCSEnergyParametrizationPCABinCalculator.h"
#include "TFCSMakeFirstPCA.h"
#include "TFCSSampleDiscovery.h"

#include "TFile.h"
#include "TH2I.h"
#include "TCanvas.h"
#include "TLegend.h"

using namespace std;


static const char USAGE[] =
    R"(Run energy parametrization

Usage:
  run_epara [<dsid>] [(-s <seed> | --seed <seed>)] [(-i <file> | --inputs <file>)] [--dsid-db <file>]
  run_epara (-h | --help)

Options:
  -h --help                   Show help screen.
  -s <seed>, --seed <seed>    Random seed [default: 42].
  -i <file>, --inputs <file>  Input samples list [default: InputSamplesList.txt].
  --dsid-db <file>            DSID DB file [default: DSID_DB.txt].
)";


//how to run on lxplus7:
//lsetup root
//.x init_epara.C+
//.x run_epara.C+

void run_epara_loop();
void run_epara();
void run_epara(int dsid, long seed, const std::string &inputs = "InputSamplesList.txt", const std::string &db = "DSID_DB.txt");
void run_epara(int dsid, long seed, const std::unique_ptr<TFCSSampleDiscovery> &samples);

TH1* get_cumul(TH1* hist);
void epara_validation_plots(TString, TString, TString, int, bool);
void epara_validation_plots_fromtree(TString, TString, TString, TString);
double runde(double x,int n);

int main(int argc, char **argv)
{
  std::map<std::string, docopt::value> args
      = docopt::docopt(USAGE, {argv + 1, argv + argc}, true);

  int dsid = args["<dsid>"].isLong() ? args["<dsid>"].asLong() : 431004;
  long seed = args["--seed"].asLong();
  std::string inputs = args["--inputs"].asString();
  std::string db = args["--dsid-db"].asString();

  run_epara(dsid, seed, inputs, db);

  return 0;
}

void run_epara_loop(long seed=42)
{
 vector<int> dsid;

 ifstream file("list_photon_131GeV.txt");
 if(!file) cout<<"file not found :O"<<endl;
 string line;
 while(getline(file, line))
 {
  stringstream linestream(line);
  string dsname;
  linestream >> dsname;
  cout<<"found dsid "<<dsname<<endl;
  dsid.push_back(atoi(dsname.c_str()));
 } // while file
 
 for(unsigned int i=0;i<dsid.size();i++) {
  run_epara(dsid[i], seed);
 }
}

void run_epara()
{
 run_epara(431004, 42);
}

void run_epara(int dsid, long seed, const std::string &inputs, const std::string &db)
{
 auto samples = std::make_unique<TFCSSampleDiscovery>(db);

 string topDir = "./output/";
 string version = "ver01";
 int npca1 = 5;
 int npca2 = 1;
 bool run_validation=true; // this will directly create the standard validation plots
 bool do_PCAclosure=true;  //this will rerun the simulated events through the 1st PCA
 
 // Create random engine
 CLHEP::TRandomEngine *randEngine = new CLHEP::TRandomEngine();
 randEngine->setSeed(seed);
 
 /////////////////////////////
 // read sample information based on DSID
 ////////////////////////////

 FCS::SampleInfo sample = samples->findSample(dsid, inputs);
 string baselabel=Form("ds%i",dsid);
 
 if (sample.location.empty()) {
   std::cout << "Error: input is empty or not found!" << std::endl;
   exit(1);
 }
 
 double sample_energy = samples->getEnergy(sample.dsid);
 int sample_pdgId     = samples->getPdgId(sample.dsid);
 double sample_mass=0;
 if (sample_pdgId == 11)  sample_mass = 0.511;
 if (sample_pdgId == 211) sample_mass = 139.6;
 double sample_Ekin=sample_energy-sample_mass;
 cout<<"sample_Ekin "<<sample_Ekin<<endl;
 
 /////////////////////////////////////////
 // form names for ouput files and directories
 ///////////////////////////////////////////
 
 system(("mkdir -p " + topDir).c_str());
 TString inputSample(Form("%s", sample.location.c_str()));
 TString pca1Filename(Form("%s%s.FirstPCA.%s.root", topDir.c_str(), baselabel.c_str(), version.c_str()));
 TString pca1AppFilename(Form("%s%s.FirstPCA_App.%s.root", topDir.c_str(), baselabel.c_str(), version.c_str()));
 TString pca2Filename(Form("%s%s.secondPCA.%s.root", topDir.c_str(), baselabel.c_str(), version.c_str()));
 TString valiFilename(Form("%s%s.eparavali.%s.root", topDir.c_str(), baselabel.c_str(), version.c_str()));
 TString valiPlotsFilename(Form("%s%s.eparavaliPlots.%s.pdf", topDir.c_str(), baselabel.c_str(), version.c_str()));
 
 /////////////////////////////////////////
 // read input sample and create first pca
 ///////////////////////////////////////////
 
 cout<<"*** Preparing to run on "<<inputSample <<" ***"<<endl;
 TChain* mychain= new TChain("FCS_ParametrizationInput");
 mychain->Add(inputSample);
 cout<<"TChain entries: "<<mychain->GetEntries()<<endl;
 
 TFCSMakeFirstPCA *myfirstPCA=new TFCSMakeFirstPCA(mychain,pca1Filename.Data());
 myfirstPCA->set_cumulativehistobins(5000);
 myfirstPCA->set_edepositcut(0.001);
 myfirstPCA->apply_etacut(0);
 myfirstPCA->use_absolute_layercut(1); //1 switches this on
 //myfirstPCA->run();
 myfirstPCA->run(randEngine);
 delete myfirstPCA;
 cout<<"TFCSMakeFirstPCA done"<<endl;
 
 TFCSApplyFirstPCA *myfirstPCA_App=new TFCSApplyFirstPCA(pca1Filename.Data());
 myfirstPCA_App->set_pcabinning(npca1,npca2);
 myfirstPCA_App->init();
 //myfirstPCA_App->run_over_chain(mychain,pca1AppFilename.Data());
 myfirstPCA_App->run_over_chain(randEngine,mychain,pca1AppFilename.Data());
 delete myfirstPCA_App;
 cout<<"TFCSApplyFirstPCA done"<<endl;
 
 delete mychain;
 
 /////////////////////////////////////////
 // create 2nd pca
 ///////////////////////////////////////////
 
 secondPCA* mysecondPCA=new secondPCA(pca1AppFilename.Data(), pca2Filename.Data());
 mysecondPCA->set_PCAbin(0);
 mysecondPCA->set_storeDetails(0);
 mysecondPCA->set_cumulativehistobins(1000);
 mysecondPCA->set_cut_maxdeviation_regression(5);
 mysecondPCA->set_cut_maxdeviation_smartrebin(0);
 mysecondPCA->set_Ntoys(5000);
 mysecondPCA->set_neurons_iteration(2,16);
 mysecondPCA->set_skip_regression(1); //if set to 1 the regression is switched off
 //mysecondPCA->run();
 mysecondPCA->run(randEngine);
 delete mysecondPCA;
 cout<<"2nd pca done"<<endl;
 
 if(run_validation)
 {
 // ************************************************** TOY simulation *********************************************************** //
  cout<<"Preparing validation histograms"<<endl;
  TFile* file1=TFile::Open(pca1AppFilename);
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
  for(unsigned int l=0;l<layer.size();l++)
   cout<<"l "<<l<<" "<<layer[l]<<endl;
  
  std::vector<TH1D*> h_output_fine;
  h_output_fine.reserve(layer.size()+2);

  for(unsigned int l=0;l<layerNr.size();l++)
   h_output_fine.emplace_back(new TH1D(Form("h_output_%s",layer[l].c_str()),Form("h_output_%s",layer[l].c_str()),5000,0.,1.));
  
  //Total E
  TTree* InputTree = (TTree*)file1->Get("tree_1stPCA");
  TreeReader* read_inputTree = new TreeReader();
  read_inputTree->SetTree(InputTree);
  double minE=InputTree->GetMinimum("energy_totalE");
  if(minE>0) minE*=0.95;
  else       minE*=1.05;
  double maxE=InputTree->GetMaximum("energy_totalE")*1.05;
  h_output_fine[layerNr.size()]=new TH1D("h_output_totalE","h_output_totalE",5000,minE,maxE);
  
  delete read_inputTree;
  
  TFCSTruthState* truth=new TFCSTruthState();
  
  TFCSPCAEnergyParametrization etest("etest","etest");
  TFile* file2 = TFile::Open(pca2Filename);
  etest.set_Ekin_nominal(sample_Ekin);

  bool check_load=etest.loadInputs(file2);
  if (!check_load) {
    std::cerr << "could not load PCA parametrization" << std::endl;
    exit(1);
  }
  cout<<"number of pca bins after load "<<etest.n_pcabins()<<" , number from firstPCA file: "<<pcabins<<endl;

  TVectorF* pcabinprob=(TVectorF*)file2->Get("PCAbinprob");
  float* prob  =pcabinprob->GetMatrixArray();
  
  float* prob_ordered=new float[pcabins+1];
  float ptot,p;
  ptot=p=0.0;
  for(int i=0;i<=pcabins;i++)
   ptot+=(float)prob[i];
  for(int i=0;i<=pcabins;i++)
  {
   p+=prob[i]/ptot;
   prob_ordered[i]=p;
  }
  
  for(int i=0;i<=pcabins;i++)
  {
   cout<<"bin "<<i<<" probability: "<<prob[i]<<" ordered: "<<prob_ordered[i]<<endl;
  }
  file2->Close();
  delete file2;
  
  //Run the loop:
  int ntoys=100000;
  //TRandom3* Random=new TRandom3(0);
  
  const TFCSExtrapolationState* extrapol=new TFCSExtrapolationState();
  //TFCSSimulationState simulstate;
  TFCSSimulationState simulstate(randEngine);

  vector<TFCSSimulationState> simulstates;
  vector<int>   randombins;
  vector<double> scalefactors;
  vector<TH1D*> cumul_simdata;
  
  TTree* tree_sim=new TTree("tree_sim","tree_sim");
  tree_sim->SetDirectory(0);
  double* simulated_efrac = new double[layerNr.size()];
  double* simulated_PCA   = new double[layer.size()];
  double  simulated_totalE;
  int     randombin;
  int     simulated_PCAbin;
  double  scalefactor;
  for(unsigned int l=0;l<layerNr.size();l++)
   tree_sim->Branch(Form("simulated_efrac_layer%i",layerNr[l]),&simulated_efrac[l],Form("simulated_efrac_layer%i/D",layerNr[l]));
  for(unsigned int l=0;l<layer.size();l++)
   tree_sim->Branch(Form("simulated_PCA_comp%i",l),&simulated_PCA[l],Form("simulated_PCA_comp%i/D",l));
  tree_sim->Branch("simulated_totalE",&simulated_totalE);
  tree_sim->Branch("randombin",&randombin);
  tree_sim->Branch("simulated_PCAbin",&simulated_PCAbin);
  tree_sim->Branch("scalefactor",&scalefactor,"scalefactor/D");
  
  for(int i=0;i<ntoys;i++)
  {
   if(i%10000==0) cout<<"Now run simulation for Toy "<<i<<endl;
   
   float searchRand = CLHEP::RandFlat::shoot(simulstate.randomEngine());
   randombin=TMath::BinarySearch(pcabins+1, prob_ordered, searchRand)+1;
   
   //cout<<"randombin "<<randombin<<endl;
   
   simulstate.set_Ebin(randombin);
   
   simulstate.set_E(sample_Ekin); //this is E_ISF
   
   etest.simulate(simulstate, truth, extrapol);
   
   for(unsigned int l=0;l<layerNr.size();l++)
    h_output_fine[l]->Fill(simulstate.Efrac(layerNr[l]));
   h_output_fine[layerNr.size()]->Fill(simulstate.E());
   
   simulstates.push_back(simulstate);
   randombins.push_back(randombin);
   scalefactors.push_back(simulstate.get_SF());
   
   if(!do_PCAclosure)
   {
    for(unsigned int l=0;l<layerNr.size();l++)
     simulated_efrac[l]=simulstate.Efrac(layerNr[l]);
    simulated_totalE=simulstate.E();
    tree_sim->Fill();
   }
  }
  
  etest.clean();
  delete extrapol;
  delete truth;
  //delete Random;
  
  if(do_PCAclosure)
  {
   cout<<"pca closure check starts"<<endl;
  
   for(unsigned int l=0;l<layer.size();l++)
   {
    TH1D* h_cumul=(TH1D*)h_output_fine[l]->Clone(Form("h_cumul_%s",layer[l].c_str()));
    for (int b=1; b<=h_cumul->GetNbinsX(); b++)
    {
      h_cumul->SetBinContent(b,h_output_fine[l]->Integral(1,b));
      h_cumul->SetBinError(b,0);
    }
    h_cumul->Scale(1.0/h_cumul->GetBinContent(h_cumul->GetNbinsX()));
    cumul_simdata.push_back(h_cumul);
   }
   
   TFCSApplyFirstPCA closurefirstPCA_App(pca1Filename.Data());
   closurefirstPCA_App.set_pcabinning(npca1,npca2);
   closurefirstPCA_App.init();
   closurefirstPCA_App.set_cumulative_energy_histos(cumul_simdata);
   
   TFCSEnergyParametrizationPCABinCalculator PCABinCalculator(closurefirstPCA_App,"test","test");
   
   for(unsigned int i=0;i<simulstates.size();i++)
   {
    //simulated_PCAbin=closurefirstPCA_App.get_PCAbin_from_simstate(simulstates[i]);
    
    PCABinCalculator.simulate(simulstates[i],truth, extrapol);
    simulated_PCAbin=PCABinCalculator.PCAbin();
    
    vector<double> PCAdata=closurefirstPCA_App.get_PCAdata_from_simstate(simulstates[i]);
    for(unsigned int l=0;l<layer.size();l++)
     simulated_PCA[l]=PCAdata[l];
    
    for(unsigned int l=0;l<layerNr.size();l++)
     simulated_efrac[l]=simulstates[i].Efrac(layerNr[l]);
    simulated_totalE=simulstates[i].E();
    randombin=randombins[i];
    scalefactor=scalefactors[i];
    tree_sim->Fill();
    
   } //loop over simulated events
   
   for(unsigned int i=0;i<cumul_simdata.size();i++)
    delete cumul_simdata[i];
   
   cout<<"at the end of the PCA closure"<<endl;
   
  } //do_PCAclosure ******************************************************************************************************************
  
  TFile* outfile=TFile::Open(valiFilename,"RECREATE");
  cout<<"writing output to file "<<outfile->GetName()<<endl;
  tree_sim->Write();
  outfile->Write();
  cout<<"outfile written"<<endl;
  
  epara_validation_plots_fromtree(pca1Filename,pca1AppFilename,valiFilename,valiPlotsFilename);
  
  for(unsigned int l=0;l<=layerNr.size();l++)
   delete h_output_fine[l];
  file1->Close();

 } //do epara validation
}


TH1* get_cumul(TH1* hist)
{
  TH1D* h_cumul=(TH1D*)hist->Clone("h_cumul");
  double sum=0;
  for(int b=1;b<=h_cumul->GetNbinsX();b++)
  {
    sum+=hist->GetBinContent(b);
    h_cumul->SetBinContent(b,sum);
  }
  return h_cumul; 
}


void epara_validation_plots_fromtree(TString firstPCAfilename, TString firstPCA_Appfilename, TString valifilename, TString valiplotsname)
{
 double GeV=1000.0;
 
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
 
 vector<TH1D*> h_g4;
 vector<TH1D*> h_sim;
 
 TFile* file_pca1=TFile::Open(firstPCAfilename);
 TTree* T_Gauss=(TTree*)file_pca1->Get("T_Gauss");
 TreeReader* read_T_Gauss = new TreeReader();
 read_T_Gauss->SetTree(T_Gauss);
 
 TFile* file_app=TFile::Open(firstPCA_Appfilename);
 vector<int> layerNr;
 TH2I* h_layer=(TH2I*)file_app->Get("h_layer");
 for(int i=1;i<=h_layer->GetNbinsY();i++)
 {
  if(h_layer->GetBinContent(1,i)) layerNr.push_back(h_layer->GetYaxis()->GetBinCenter(i));
 }
 
 TTree* tree_g4=(TTree*)file_app->Get("tree_1stPCA");
 
 TFile* file_vali=TFile::Open(valifilename);
 TTree* tree_sim=(TTree*)file_vali->Get("tree_sim");
 
 vector<TH1D*> h_efrac_g4;
 vector<TH1D*> h_efrac_sim;
 for(unsigned int l=0;l<layerNr.size();l++)
 {
  TH1D* hist=new TH1D(Form("h_efrac_layer%i_g4",layerNr[l]),Form("h_efrac_layer%i_g4",layerNr[l]),200,-0.1,1.1);
  hist->GetXaxis()->SetTitle(Form("E fraction in layer %i (%s)",layerNr[l],lname[layerNr[l]].c_str()));
  h_efrac_g4.push_back(hist);
  TH1D* hist2=new TH1D(Form("h_efrac_layer%i_sim",layerNr[l]),Form("h_efrac_layer%i_sim",layerNr[l]),200,-0.1,1.1);
  h_efrac_sim.push_back(hist2);
 }
 double emin=tree_g4->GetMinimum("energy_totalE"); emin*=1.2;
 double emax=tree_g4->GetMaximum("energy_totalE"); emax*=1.05;
 TH1D* h_totalE_g4=new TH1D("h_totalE_g4","h_totalE_g4",200,emin/GeV,emax/GeV);
 h_totalE_g4->GetXaxis()->SetTitle("total energy [GeV]");
 TH1D* h_totalE_sim=new TH1D("h_totalE_sim","h_totalE_sim",200,emin/GeV,emax/GeV);
 
 vector<TH1D*> h_PC_g4;
 vector<TH1D*> h_PC_sim;
 double pcmax=tree_sim->GetMaximum("simulated_PCA_comp0")*1.2;
 if(pcmax<5) pcmax=5;
 for(unsigned int l=0;l<=layerNr.size();l++)
 {
  TH1D* hist=new TH1D(Form("h_PC%i_g4",l),Form("h_PC%i_g4",l),200,-pcmax,pcmax);
  hist->GetXaxis()->SetTitle(Form("PC %i",l));
  h_PC_g4.push_back(hist);
  TH1D* hist2=new TH1D(Form("h_PC%i_sim",l),Form("h_PC%i_sim",l),200,-pcmax,pcmax);
  h_PC_sim.push_back(hist2);
 }
 
 //g4:
 TreeReader* read_tree_g4 = new TreeReader();
 read_tree_g4->SetTree(tree_g4);
 for(int event=0;event<read_tree_g4->GetEntries();event++)
 {
  read_tree_g4->GetEntry(event);
  h_totalE_g4->Fill(read_tree_g4->GetVariable("energy_totalE")/GeV);
  for(unsigned int l=0;l<layerNr.size();l++)
  {
   double frac=read_tree_g4->GetVariable(Form("energy_layer%i",layerNr[l]));
   h_efrac_g4[l]->Fill(frac);
  }
 }
 
 double sfmax=tree_sim->GetMaximum("scalefactor");
 if(sfmax>3) sfmax=2;
 double sfmin=tree_sim->GetMinimum("scalefactor");
 if(sfmax<0) sfmax=0;
 TH1D* h_sf=new TH1D("h_sf","h_sf",200,sfmin,sfmax);
 h_sf->GetXaxis()->SetTitle("Scalefactor");
 
 //sim:
 TreeReader* read_tree_sim = new TreeReader();
 read_tree_sim->SetTree(tree_sim);
 int count_match=0;
 for(int event=0;event<read_tree_sim->GetEntries();event++)
 {
  read_tree_sim->GetEntry(event);
  h_totalE_sim->Fill(read_tree_sim->GetVariable("simulated_totalE")/GeV);
  for(unsigned int l=0;l<layerNr.size();l++)
   h_efrac_sim[l]->Fill(read_tree_sim->GetVariable(Form("simulated_efrac_layer%i",layerNr[l])));
  for(unsigned int l=0;l<=layerNr.size();l++)
   h_PC_sim[l]->Fill(read_tree_sim->GetVariable(Form("simulated_PCA_comp%i",l)));
  if(read_tree_sim->GetVariable("randombin")==read_tree_sim->GetVariable("simulated_PCAbin"))
   count_match++;
  h_sf->Fill(read_tree_sim->GetVariable("scalefactor"));
 }
 
 for(int event=0;event<read_T_Gauss->GetEntries();event++)
 {
  read_T_Gauss->GetEntry(event);
  for(unsigned int l=0;l<=layerNr.size();l++)
   h_PC_g4[l]->Fill(read_T_Gauss->GetVariable(Form("data_PCA_comp%i",l)));
 }
 
 string title=h_PC_g4[0]->GetXaxis()->GetTitle();
 h_PC_g4[0]->GetXaxis()->SetTitle(Form("Matched bins: %.1f%%   %s",(double)count_match/(double)read_tree_sim->GetEntries()*100.0,title.c_str()));
 
 //add all the histos to the list
 h_g4=h_efrac_g4;
 h_g4.push_back(h_totalE_g4);
 h_g4.push_back(h_PC_g4[0]);
 h_g4.push_back(h_PC_g4[1]);
 
 h_sim=h_efrac_sim;
 h_sim.push_back(h_totalE_sim);
 h_sim.push_back(h_PC_sim[0]);
 h_sim.push_back(h_PC_sim[1]);
 
 for(unsigned int i=0;i<h_sim.size();i++)
 {
  //h_sim[i]->Scale(h_g4[i]->Integral(0,h_g4[i]->GetNbinsX()+1)/h_sim[i]->Integral(0,h_sim[i]->GetNbinsX()+1));
  h_sim[i]->Scale(h_g4[i]->Integral()/h_sim[i]->Integral());
  h_sim[i]->SetLineColor(2);
  h_sim[i]->SetLineStyle(2);
  h_sim[i]->SetMarkerColor(2);
  h_sim[i]->SetMarkerStyle(24);
  h_g4[i]->SetLineWidth(1);
  h_sim[i]->SetLineWidth(1);
  h_g4[i]->SetMarkerSize(1.0);
  h_sim[i]->SetMarkerSize(1.0);
 }
 
 for(unsigned int i=0;i<h_sim.size();i++)
 {
  double ymax=h_g4[i]->GetBinContent(h_g4[i]->GetMaximumBin());
  if(h_sim[i]->GetBinContent(h_sim[i]->GetMaximumBin())>ymax) ymax=h_sim[i]->GetBinContent(h_sim[i]->GetMaximumBin());
  ymax*=1.2;
  double ymax_log=ymax*2.0;
  TH1D* h_g4_log=(TH1D*)h_g4[i]->Clone("h_g4_log");
  h_g4_log->GetYaxis()->SetRangeUser(0.1,ymax_log);
  TH1D* h_sim_log=(TH1D*)h_sim[i]->Clone("h_sim_log");
  
  TLegend* leg=new TLegend(0.2,0.9,0.8,0.94);
  leg->SetNColumns(2);
  leg->AddEntry(h_g4_log,"G4","lpe");
  leg->AddEntry(h_sim_log,"Sim","lpe");
  leg->SetBorderSize(0);
  leg->SetFillStyle(0);
  
  TCanvas* can=new TCanvas("can","can",0,0,1200,600);
  can->Divide(2);
  can->cd(1);
  h_g4[i]->Draw("e");
  h_g4[i]->GetYaxis()->SetRangeUser(0,ymax);
  h_sim[i]->Draw("esame");
  leg->Draw();
  
  can->cd(2);
  h_g4_log->Draw("e");
  h_sim_log->Draw("esame");
  can->cd(2)->SetLogy();
  leg->Draw();
  
  if(i==0) can->Print(Form("%s(",valiplotsname.Data()));
  else can->Print(Form("%s",valiplotsname.Data()));
  
  delete can;
  delete h_g4_log;
  delete h_sim_log;
 }
 
 TCanvas* can=new TCanvas("can","can",0,0,1200,600);
 h_sf->Draw("hist");
 can->Print(Form("%s)",valiplotsname.Data()));
 
 file_app->Close();
 file_vali->Close();
 file_pca1->Close();
 
}

/*
void epara_validation_plots(TString firstPCAfilename, TString valifilename, TString valiplotsname, int dsid, bool do_PCAclosure)
{
  
  cout<<"now producing standard validation plots"<<endl;
  
  int col_para=2;
  
  //get layers:
  TFile* file_pca1=TFile::Open(firstPCAfilename);
  TH2I* h_layer=(TH2I*)file_pca1->Get("h_layer");
  h_layer->SetDirectory(0);
  file_pca1->Close();
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
  
  TFile* file=TFile::Open(valifilename);
  if(!file) cout<<"VALIDATION FILE NOT OPEN"<<endl;
  int use_autozoom=1;
  
  for(unsigned int l=0;l<layer.size();l++)
  {
   cout<<"now do plots for layer "<<layer[l]<<endl;
   
   double min,max,rmin,rmax;
   TH1D* h_output_lin;
   TH1D* h_input_lin;
   if(use_autozoom)
   {
   	h_output_lin=(TH1D*)file->Get(Form("h_output_zoom_%s",layer[l].c_str())); h_output_lin->SetName("h_output_lin");
   	h_input_lin =(TH1D*)file->Get(Form("h_input_zoom_%s",layer[l].c_str()));  h_input_lin->SetName("h_input_lin");
   }
   else
   {
   	h_output_lin=(TH1D*)file->Get(Form("h_output_%s",layer[l].c_str())); h_output_lin->SetName("h_output_lin");
   	h_input_lin =(TH1D*)file->Get(Form("h_input_%s",layer[l].c_str()));  h_input_lin->SetName("h_input_lin");
   }
   
   //linear:
   double ymax=1000;
   double chi2= h_input_lin->Chi2Test(h_output_lin,"UW");
   h_input_lin->SetMarkerSize(1.1);
   h_output_lin->SetMarkerSize(1.1);
   h_input_lin->SetMarkerStyle(24);
   
   h_input_lin->SetLineWidth(1);
   h_output_lin->SetLineWidth(1);
   h_output_lin->SetFillColor(col_para);
   h_output_lin->SetLineColor(col_para);
   h_output_lin->SetMarkerColor(col_para);
   h_output_lin->Scale(h_input_lin->Integral()/h_output_lin->Integral());
   h_input_lin->GetXaxis()->SetNdivisions(506,kTRUE);
   ymax=h_input_lin->GetBinContent(h_input_lin->GetMaximumBin());
   h_input_lin->GetYaxis()->SetRangeUser(0,ymax*1.4);
   h_input_lin->GetYaxis()->SetTitle("a.u.");
   h_input_lin->GetXaxis()->SetTitle(title[l].c_str());
   
   //log:
   TH1D* h_output_log=(TH1D*)h_output_lin->Clone("h_output_log");
   TH1D* h_input_log=(TH1D*)h_input_lin->Clone("h_input_log");
   h_input_log->GetYaxis()->SetRangeUser(0.1,ymax*5.0);
   h_input_log->GetYaxis()->SetTitle("a.u.");
   
   //cumulative:
   TH1D* h_output_cumul=(TH1D*)get_cumul(h_output_lin); h_output_cumul->SetName("h_output_cumul");
   TH1D* h_input_cumul =(TH1D*)get_cumul(h_input_lin);  h_input_cumul->SetName("h_input_cumul");
   double sf=h_input_cumul->GetBinContent(h_input_cumul->GetNbinsX());
   h_output_cumul->Scale(1.0/sf);
   h_input_cumul->Scale(1.0/sf);
   h_input_cumul->GetYaxis()->SetRangeUser(0,1.2);
   h_input_cumul->GetYaxis()->SetTitle("a.u. cumulative");

   TCanvas* can=new TCanvas("can","can",0,0,1600,600);
   can->Divide(3,1);
   can->cd(1);
   h_input_lin->Draw("e");
   h_output_lin->Draw("esame");
   h_input_lin->Draw("esame");
    
   TLegend* leg=new TLegend(0.2,0.85,0.8,0.93);
   leg->SetBorderSize(0);
   leg->SetFillStyle(0);
   leg->SetHeader(Form("%i , Chi2 %.2f",dsid,chi2));
   leg->SetNColumns(2);
   leg->AddEntry(h_output_lin,"Parametrisation","lpe");
   leg->AddEntry(h_input_lin,"G4 Input","lpe");
   leg->Draw();
    
   can->cd(2);
   h_input_log->Draw("e");
   h_output_log->Draw("esame");
   h_input_log->Draw("esame");
   can->cd(2)->SetLogy();

   can->cd(3);
   h_input_cumul->Draw("e");
   h_output_cumul->Draw("esame");
   h_input_cumul->Draw("esame");
   
   can->cd(1)->RedrawAxis();
   can->cd(2)->RedrawAxis();
   can->cd(3)->RedrawAxis();
    
   //This produces a merged pdf:
   if(l==0)
    can->Print(Form("%s(",valiplotsname.Data()));
   else
   {
    if(l==layer.size()-1 && !do_PCAclosure)
     can->Print(Form("%s)",valiplotsname.Data()));
    else
     can->Print(Form("%s",valiplotsname.Data()));
   }
    
   can->Close();
   delete leg;
   delete can;
   delete h_output_lin;
   delete h_input_lin;
   
  } //for layer
  
  if(do_PCAclosure)
  {
   cout<<"now store the PCA closure plots"<<endl;
   TH1D* h_comp0=(TH1D*)file->Get("h_comp0");
   TH1D* h_comp0_closure=(TH1D*)file->Get("h_comp0_closure");
   h_comp0_closure->Scale(h_comp0->Integral()/h_comp0_closure->Integral());
   TH1D* h_match=(TH1D*)file->Get("h_match");
   TCanvas* can=new TCanvas("can","can",0,0,1200,600);
   can->Divide(2);
   can->cd(1);
   h_comp0->SetLineWidth(1);
   h_comp0_closure->SetLineWidth(1);
   h_comp0->SetMarkerSize(1.1);
   h_comp0_closure->SetMarkerSize(1.1);
   h_comp0->Draw("hist");
   h_comp0->GetXaxis()->SetTitle("leading component");
   double max=h_comp0->GetBinContent(h_comp0->GetMaximumBin());
   if(h_comp0_closure->GetBinContent(h_comp0_closure->GetMaximumBin())>max) max=h_comp0_closure->GetBinContent(h_comp0_closure->GetMaximumBin());
   h_comp0->GetYaxis()->SetRangeUser(0,max*1.2);
   h_comp0_closure->SetLineColor(2);
   h_comp0_closure->SetLineStyle(2);
   h_comp0_closure->SetMarkerColor(2);
   h_comp0_closure->SetMarkerStyle(24);
   h_comp0_closure->Draw("histsame");
   TLegend* leg=new TLegend(0.2,0.75,0.4,0.9);
   leg->SetLineStyle(0);
   leg->SetBorderSize(0);
   leg->AddEntry(h_comp0,"FirstPCA","lpe");
   leg->AddEntry(h_comp0_closure,"Closure","lpe");
   leg->Draw();
   can->cd(2);
   h_match->Draw();
   can->Print(Form("%s)",valiplotsname.Data()));
   delete leg;
   delete can;
   delete h_comp0;
   delete h_comp0_closure;
  }
  
  delete h_layer;
  file->Close();
 
}

*/

double runde(double x,int n)
{
 double wert;
 x*=(double)(pow(10.,(double)n));
 if((x-0.5)<floor(x))
  wert=floor(x)/(double)(pow(10.0,(double)n));
 else
  wert=ceil(x)/(double)(pow(10.0,(double)n));
 
 return wert;
}
