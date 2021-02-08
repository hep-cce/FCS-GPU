/*
  Copyright (C) 2002-2017 CERN for the benefit of the ATLAS collaboration
*/

{
  if(1==0)
  {
    // Needs athena environment setup and ISF_FastCaloSimParametrization package compiled.
    // Uses the root interface library created in the compilation of ISF_FastCaloSimParametrization
    gSystem->Load("libISF_FastCaloSimParametrizationLib.so");
  }
  else
  {
    gSystem->AddIncludePath(" -I.. ");
    
    gROOT->LoadMacro("../Root/IntArray.cxx+");
    gROOT->LoadMacro("../Root/TreeReader.cxx+");
    gROOT->LoadMacro("../Root/TFCS1DFunction.cxx+");
    gROOT->LoadMacro("../Root/TFCS1DFunctionRegression.cxx+");
    gROOT->LoadMacro("../Root/TFCS1DFunctionRegressionTF.cxx+");
    gROOT->LoadMacro("../Root/TFCS1DFunctionHistogram.cxx+");
    gROOT->LoadMacro("../Root/TFCSFunction.cxx+");
    gROOT->LoadMacro("../Root/TFCSExtrapolationState.cxx+");
    gROOT->LoadMacro("../Root/TFCSTruthState.cxx+");
    gROOT->LoadMacro("../Root/TFCSSimulationState.cxx+");
    gROOT->LoadMacro("../Root/TFCSParametrizationBase.cxx+");
    gROOT->LoadMacro("../Root/TFCSParametrization.cxx+");
    gROOT->LoadMacro("../Root/TFCSEnergyParametrization.cxx+");
    gROOT->LoadMacro("../Root/TFCSPCAEnergyParametrization.cxx+");
    gROOT->LoadMacro("../Root/EnergyParametrizationValidation.cxx+");
/*
    gROOT->LoadMacro("../Root/TFCSLateralShapeParametrization.cxx+");
    gROOT->LoadMacro("../Root/TFCSNNLateralShapeParametrization.cxx+");
    gROOT->LoadMacro("../Root/TFCSSimpleLateralShapeParametrization.cxx+");
*/
  }
  
  string samplename="pions";
  //string samplename="pions_s2864";
  //string samplename="pions_s2865";
  
  int setbin=0;
  cout<<"PCA bin (-1 if random)? "<<endl;
  cin>>setbin;
  
  //Prepare the Histograms
  cout<<"Preparing validation histograms"<<endl;
  TFile* file1=TFile::Open(Form("output/firstPCA_%s.root",samplename.c_str()));
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
  
  int nbins=100;
  TH1D* h_input[layer.size()+2];
  TH1D* h_output[layer.size()+2];
  for(unsigned int l=0;l<layerNr.size();l++)
  {
   h_input[l]=new TH1D(Form("h_input_%s",layer[l].c_str()),Form("h_input_%s",layer[l].c_str()),nbins,0,1);
   h_output[l]=new TH1D(Form("h_output_%s",layer[l].c_str()),Form("h_output_%s",layer[l].c_str()),nbins,0,1);
  }
  //Total E
  TTree* InputTree = (TTree*)file1->Get("tree_1stPCA");
  TreeReader* read_inputTree = new TreeReader();
  read_inputTree->SetTree(InputTree);

  double minE=InputTree->GetMinimum("energy_totalE");
  double maxE=InputTree->GetMaximum("energy_totalE");
  cout<<"************************"<<endl;
  cout<<"minE "<<minE<<" maxE "<<maxE<<endl;
  h_input[layerNr.size()] =new TH1D("h_input_totalE","h_input_totalE",nbins,minE,maxE);
  h_output[layerNr.size()]=new TH1D("h_output_totalE","h_output_totalE",nbins,minE,maxE);
  //Fractions:
  h_input[layer.size()]   =new TH1D("h_input_sumfractions_elmag","h_input_sumfractions_elmag",nbins,-1,2);
  h_output[layer.size()]  =new TH1D("h_output_sumfractions_elmag","h_output_sumfractions_elmag",nbins,-1,2);
  h_input[layer.size()+1] =new TH1D("h_input_sumfractions_had","h_input_sumfractions_had",nbins,-1,2);
  h_output[layer.size()+1]=new TH1D("h_output_sumfractions_had","h_output_sumfractions_had",nbins,-1,2);
  
  //Fill the Input Histograms:
  vector<int> elmag; //0-8
  for(int e=0;e<=8;e++) elmag.push_back(e);
  vector<int> had;   //9-24
  for(int h=9;h<25;h++) had.push_back(h);
  
  cout<<"Now fill input histograms"<<endl;
  for(int event=0;event<read_inputTree->GetEntries();event++)
  {
   read_inputTree->GetEntry(event);
   if(setbin<0 || (setbin>=0 && read_inputTree->GetVariable("firstPCAbin")==setbin))
   {
   double sum_fraction_elmag=0.0;
   double sum_fraction_had=0.0;
   double data = read_inputTree->GetVariable("energy_totalE");
   h_input[layerNr.size()]->Fill(data);
   for(unsigned int l=0;l<layerNr.size();l++)
   {
   	double data = read_inputTree->GetVariable(Form("energy_%s",layer[l].c_str()));
   	h_input[l]->Fill(data);
   	
   	int is_elmag,is_had;
   	is_elmag=is_had=0;
    
    for(int e=0;e<elmag.size();e++)
    { 	  if(elmag[e]==layerNr[l]) is_elmag=1;   }
    for(int h=0;h<had.size();h++)
    { 	  if(had[h]==layerNr[l]) is_had=1;   }
    if(is_elmag) sum_fraction_elmag+=data;
    if(is_had)   sum_fraction_had+=data;
   }
   h_input[layerNr.size()+1]->Fill(sum_fraction_elmag);
   h_input[layerNr.size()+2]->Fill(sum_fraction_had);
   }
  }
  
  TH1D* h_randombin=new TH1D("h_randombin","h_randombin",pcabins,-0.5,pcabins-0.5);
  
  //Run the loop:
  int ntoys=5000;
  TRandom3* Random=new TRandom3();
  Random->SetSeed(0);
  const TFCSTruthState* truth=new TFCSTruthState();
  const TFCSExtrapolationState* extrapol=new TFCSExtrapolationState();
  for(int i=0;i<ntoys;i++)
  {
   //if(i%100==0) 
   	cout<<"Now run simulation for Toy "<<i<<endl;
   
   double uniform=Random->Uniform(1);
   int randombin=0;
   for(int n=0;n<pcabins;n++)
   {
    if(uniform>n*1.0/(double)pcabins && uniform<(n+1.)*1.0/(double)pcabins)
     randombin=n;
   }
   if(setbin>=0)
    randombin=setbin;
   h_randombin->Fill(randombin);
   
   cout<<"call etest"<<endl;
     
   TFCSPCAEnergyParametrization* etest=new TFCSPCAEnergyParametrization("etest","etest");
   TFile* file2;
   if(setbin!=-1) file2=TFile::Open(Form("output/secondPCA_%s_bin%i.root",samplename.c_str(),randombin));
   else           file2=TFile::Open(Form("output/secondPCA_%s.root",samplename.c_str()));
   if(setbin!=-1) etest->loadInputs(file2);
   else           etest->loadInputs(file2,randombin);
   file2->Close();
   delete file2;
   
      
   TFCSSimulationState simulstate;
   simulstate.set_Ebin(randombin);
   
   
   cout<<"simulate"<<endl;
   etest->simulate(simulstate, truth, extrapol);
   delete etest;
   
   //fill the Histograms:
   double sum_fraction_elmag=0.0;
   double sum_fraction_had=0.0;
   
   for(int s=0;s<30;s++)
   {
   	int is_elmag,is_had;
   	is_elmag=is_had=0;
   	for(unsigned int l=0;l<layerNr.size();l++)
    {
     if(s==layerNr[l])
     {
   	  h_output[l]->Fill(simulstate.E(s));
      for(int e=0;e<elmag.size();e++)
      { 	  if(elmag[e]==layerNr[l]) is_elmag=1;   }
      for(int h=0;h<had.size();h++)
      { 	  if(had[h]==layerNr[l]) is_had=1;   }
      if(is_elmag) sum_fraction_elmag+=simulstate.E(s);
      if(is_had)   sum_fraction_had+=simulstate.E(s);
     }
    }
   }
   h_output[layerNr.size()]->Fill(simulstate.E());
   h_output[layerNr.size()+1]->Fill(sum_fraction_elmag);
   h_output[layerNr.size()+2]->Fill(sum_fraction_had);
   
  } //loop over toys
  
  cout<<"Now make plots"<<endl;
  
  vector<string> name;
  vector<string> title;
  for(unsigned int l=0;l<layer.size()-1;l++)
  {
   name.push_back(layer[l].c_str());
   title.push_back(Form("E fraction in Layer %i",layerNr[l]));
  }
  name.push_back("totalE");
  name.push_back("sumfraction_elmag");
  name.push_back("sumfraction_had");
  title.push_back("total E [MeV]");
  title.push_back("Sum of E fractions in elmag. layers");
  title.push_back("Sum of E fractions in had. layers");
  
  for(unsigned int l=0;l<layer.size()+2;l++)
  {
   //TCanvas* can=new TCanvas(Form("can_%i",l),Form("can_%i",l),0,0,1600,600);
   TCanvas* can=new TCanvas("can","can",0,0,1600,600);
   can->Divide(3,1);
   can->cd(1); //linear scale
   double min,max,rmin,rmax;

   int use_autozoom=1;
   TH1D* h_output_lin;
   TH1D* h_input_lin;
   if(use_autozoom)
   {
    EnergyParametrizationValidation::autozoom(h_input[l],h_output[l],min,max,rmin,rmax);
    h_output_lin=EnergyParametrizationValidation::refill(h_output[l],min,max,rmin,rmax);  h_output_lin->SetName("h_output_lin");
    h_input_lin=EnergyParametrizationValidation::refill(h_input[l],min,max,rmin,rmax); h_input_lin->SetName("h_input_lin");
   }
   else
   {
   	h_output_lin=(TH1D*)h_output[l]->Clone("h_output_lin");
   	h_input_lin=(TH1D*)h_input[l]->Clone("h_input_lin");
   }
   
   double kolmo=h_input[l]->KolmogorovTest(h_output[l]);
   double chi2=h_input[l]->Chi2Test(h_output[l],"UW");

   h_input_lin->SetMarkerSize(1.0);
   h_input_lin->SetLineWidth(0.1);
   h_output_lin->SetLineWidth(0.1);
   h_output_lin->SetFillColor(7);
   
   h_output_lin->Scale(h_input[l]->Integral()/h_output_lin->Integral());
   h_input_lin->Draw("e");
   h_input_lin->GetXaxis()->SetNdivisions(504,kFALSE);
   double ymax=h_input_lin->GetBinContent(h_input_lin->GetMaximumBin());
   h_input_lin->GetYaxis()->SetRangeUser(0,ymax*1.4);
   h_input_lin->GetYaxis()->SetTitle("Linear");
   h_input_lin->GetXaxis()->SetTitle(title[l].c_str());
   h_output_lin->Draw("histsame");
   h_input_lin->Draw("esame");
   
   TLegend* leg=new TLegend(0.65,0.82,0.99,0.93);
   leg->SetBorderSize(0);
   leg->SetFillStyle(0);
   leg->SetHeader(Form("KS: %.2f, Chi2: %.2f",kolmo,chi2));
   leg->AddEntry(h_output_lin,"Parametrisation","f");
   leg->AddEntry(h_input_lin,"G4 Input","lpe");
   leg->Draw();
  
   can->cd(2);
   TH1D* h_output_log=(TH1D*)h_output_lin->Clone("h_output_log");
   TH1D* h_input_log=(TH1D*)h_input_lin->Clone("h_input_log");
   h_input_log->Draw("e");
   h_input_log->GetYaxis()->SetRangeUser(0.1,ymax*5.0);
   h_input_log->GetYaxis()->SetTitle("Log");
   h_output_log->Draw("histsame");
   h_input_log->Draw("esame");
   can->cd(2)->SetLogy();
   TLegend* leg2=new TLegend(0.65,0.82,0.99,0.93);
   leg2->SetBorderSize(0);
   leg2->SetFillStyle(0);
   leg2->SetHeader(Form("KS: %.2f, Chi2: %.2f",kolmo,chi2));
   leg2->AddEntry(h_output_lin,"Parametrisation","f");
   leg2->AddEntry(h_input_lin,"G4 Input","lpe");
   leg2->Draw();
   
   can->cd(3);
   TH1D* h_output_cumul=(TH1D*)TFCS1DFunction::get_cumul(h_output_lin); h_output_cumul->SetName("h_output_cumul");
   TH1D* h_input_cumul =(TH1D*)TFCS1DFunction::get_cumul(h_input_lin);  h_input_cumul->SetName("h_input_cumul");
   double sf=h_input_cumul->GetBinContent(h_input_cumul->GetNbinsX());
   h_output_cumul->Scale(1.0/sf);
   h_input_cumul->Scale(1.0/sf);
   h_input_cumul->Draw("e");
   h_input_cumul->GetYaxis()->SetRangeUser(0,1.2);
   h_input_cumul->GetYaxis()->SetTitle("Cumulative");
   h_output_cumul->Draw("histsame");
   h_input_cumul->Draw("esame");
   TLegend* leg3=new TLegend(0.19,0.82,0.53,0.93);
   leg3->SetBorderSize(0);
   leg3->SetFillStyle(0);
   leg3->SetHeader(Form("KS: %.2f, Chi2: %.2f",kolmo,chi2));
   leg3->AddEntry(h_output_lin,"Parametrisation","f");
   leg3->AddEntry(h_input_lin,"G4 Input","lpe");
   leg3->Draw();
   
   can->cd(1)->RedrawAxis();
   can->cd(2)->RedrawAxis();
   can->cd(3)->RedrawAxis();
   
   if(setbin>=0)
    can->Print(Form("plots/%s_%s_bin%i.pdf",samplename.c_str(),name[l].c_str(),setbin));
   else
   	can->Print(Form("plots/%s_%s.pdf",samplename.c_str(),name[l].c_str()));
   can->Close();
   
  } //for layer
  
  //TCanvas* can=new TCanvas("can","can",0,0,800,600);  h_randombin->Draw();
  
}

