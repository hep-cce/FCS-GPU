/*
  Copyright (C) 2002-2021 CERN for the benefit of the ATLAS collaboration
*/

void P2X(TVectorD* SigmaValues, TVectorD* MeanValues, TMatrixD *EV, int gNVariables, double *p, double *x, int nTest);
double get_inverse(double rnd,TH1D*);
double non_linear(double y1,double y2,double x1,double x2,double y);

void check_correlations()
{
 
 //open file
 //loop on the tree_gauss
 //plot correlations: pca transformed data in 2D plots
 
 double pc0_cut=1.132;
 
 int dsid=433859;
 dsid=431004;
 
 TFile* file=TFile::Open(Form("output/ds%i.FirstPCA.ver01.root",dsid));
 TTree* T_Gauss=(TTree*)file->Get("T_Gauss");
 
 vector<int> comps;
 //n*(n-1)/2
 
 int n=6;
 if(dsid==433859) n=6;
 if(dsid==431004) n=5;
 
 vector<string> layer;
 if(dsid==433859)
 {
  layer.push_back("layer6");
  layer.push_back("layer7");
  layer.push_back("layer8");
  layer.push_back("layer9");
  layer.push_back("layer21");
  layer.push_back("totalE");
 }
 if(dsid==431004)
 {
  layer.push_back("layer0");
  layer.push_back("layer1");
  layer.push_back("layer2");
  layer.push_back("layer3");
  layer.push_back("layer12");
  layer.push_back("totalE");
 }
 
 for(int i=0;i<=n;i++)
  comps.push_back(i);
 
 vector<string> hname;
 vector<string> hname_layer;
 for(int a=0;a<comps.size();a++)
 {
  for(int b=0;b<comps.size();b++)
  {
   if(a!=b)
   {
    int found=0;
    for(int d=0;d<hname.size();d++)
    {
     if(Form("%i_%i",comps[b],comps[a])==hname[d])
     {
      found=1;
      d=hname.size();
     }
    }
    if(!found)
    {
     hname.push_back(Form("%i_%i",comps[a],comps[b]));
     hname_layer.push_back(Form("%s_%s",layer[a].c_str(),layer[b].c_str()));
    }
   }
  }
 }
 
 cout<<"size "<<hname.size()<<" before "<<hname_layer.size()<<endl;
 
 vector<TH2D*> h_corr;
 vector<TH2D*> h_corr_before;
 vector<TH1D*> h_PCA_cut;
 vector<TH1D*> h_Gauss_cut;
 
 for(int i=0;i<hname.size();i++)
 {
  cout<<i<<" "<<hname[i]<<endl;
  string name=Form("h_corr_%s",hname[i].c_str());
  TH2D* hist=new TH2D(name.c_str(),name.c_str(),100,-6,6,100,-6,6);
  h_corr.push_back(hist);
  
  string name2=Form("h_corr_layer_%s",hname_layer[i].c_str());
  TH2D* hist2=new TH2D(name2.c_str(),name2.c_str(),100,-6,6,100,-6,6);
  h_corr_before.push_back(hist2);
 }
 
 for(unsigned int l=0;l<comps.size();l++)
 {
  TH1D* hist=new TH1D(Form("h_PCA_cut_comp%i",comps[l]),Form("h_PCA_cut_comp%i",comps[l]),100,-6,6);
  h_PCA_cut.push_back(hist);
 }
 
 for(unsigned int l=0;l<layer.size();l++)
 {
  TH1D* hist=new TH1D(Form("h_Gauss_cut_%s",layer[l].c_str()),Form("h_Gauss_cut_%s",layer[l].c_str()),100,-6,6);
  h_Gauss_cut.push_back(hist);
 }
 
 vector<double> PCA;
 for(unsigned int l=0;l<comps.size();l++)
  PCA.push_back(0.0);
 
 vector<double> Gauss;
 for(unsigned int l=0;l<layer.size();l++)
  Gauss.push_back(0.0);
 
 TreeReader* read_TGaus = new TreeReader();
 read_TGaus->SetTree(T_Gauss);
 for(int event=0;event<read_TGaus->GetEntries();event++)
 {
  //cout<<"event "<<event<<endl;
  read_TGaus->GetEntry(event);
  for(unsigned int l=0;l<comps.size();l++)
   PCA[l] = read_TGaus->GetVariable(Form("data_PCA_comp%i",comps[l]));

  for(unsigned int l=0;l<layer.size();l++)
   Gauss[l] = read_TGaus->GetVariable(Form("data_Gauss_%s",layer[l].c_str()));
  
  for(int i=0;i<hname.size();i++)
  {
   //cout<<"i "<<i<<endl;
   char *c = new char[hname[i].length() + 1];
   strcpy(c, hname[i].c_str());
   int a=int(c[0]-'0');
   int b=int(c[2]-'0');
   h_corr[i]->Fill(PCA[a],PCA[b]);
   h_corr_before[i]->Fill(Gauss[a],Gauss[b]);
  }
  
  /*
  if(PCA[0]>pc0_cut)
  {
   for(unsigned int l=0;l<comps.size();l++)
    h_PCA_cut[l]->Fill(PCA[l]);
   for(int l=0;l<layer.size();l++)
    h_Gauss_cut[l]->Fill(Gauss[l]);
  }
 */
 } //event loop
 
 //make plots:
 gStyle->SetPalette(70);
 for(int i=0;i<hname.size();i++)
 {
  char *c = new char[hname[i].length() + 1];
  strcpy(c, hname[i].c_str());
  TCanvas* can=new TCanvas("can","can",0,0,800,600);
  h_corr[i]->Draw("boxcolz");
  double cfactor=h_corr[i]->GetCorrelationFactor();
  h_corr[i]->GetXaxis()->SetTitle(Form("Correlation: %.3f     PC %c",cfactor,c[0]));
  h_corr[i]->GetYaxis()->SetTitle(Form("PC %c",c[2]));
  if(i==0) can->Print(Form("corr_ds%i.pdf(",dsid));
  if(i==hname.size()-1) can->Print(Form("corr_ds%i.pdf)",dsid));
  if(i>0 && i<hname.size()-1) can->Print(Form("corr_ds%i.pdf",dsid));
  delete can;
 }
 for(int i=0;i<hname_layer.size();i++)
 {
  //decompose the name to get back the layers:
  stringstream ss(hname_layer[i]);
  string segment;
  vector<string> seglist;
  while(std::getline(ss, segment, '_'))
   seglist.push_back(segment);
  
  TCanvas* can=new TCanvas("can","can",0,0,800,600);
  h_corr_before[i]->Draw("boxcolz");
  double cfactor_before=h_corr_before[i]->GetCorrelationFactor();
  h_corr_before[i]->GetXaxis()->SetTitle(Form("Correlation: %.3f     %s",cfactor_before,seglist[0].c_str()));
  h_corr_before[i]->GetYaxis()->SetTitle(Form("%s",seglist[1].c_str()));
  if(i==0) can->Print(Form("corr_before_ds%i.pdf(",dsid));
  if(i==hname_layer.size()-1) can->Print(Form("corr_before_ds%i.pdf)",dsid));
  if(i>0 && i<hname_layer.size()-1) can->Print(Form("corr_before_ds%i.pdf",dsid));
  delete can;
 }
 
 /*
 for(int i=0;i<h_PCA_cut.size();i++)
 {
  TCanvas* can=new TCanvas("can","can",0,0,800,600);
  h_PCA_cut[i]->Draw("hist");
  h_PCA_cut[i]->GetXaxis()->SetTitle(Form("PC%i for PC0>1.132",comps[i]));
  if(i==0) can->Print(Form("PCA_cut_ds%i.pdf(",dsid));
  if(i==h_PCA_cut.size()-1) can->Print(Form("PCA_cut_ds%i.pdf)",dsid));
  if(i>0 && i<h_PCA_cut.size()-1) can->Print(Form("PCA_cut_ds%i.pdf",dsid));
  delete can;
 }
 
 for(int i=0;i<h_Gauss_cut.size();i++)
 {
  TCanvas* can=new TCanvas("can","can",0,0,800,600);
  h_Gauss_cut[i]->Draw("hist");
  h_Gauss_cut[i]->GetXaxis()->SetTitle(Form("%s for PC0>1.132",layer[i].c_str()));
  if(i==0) can->Print(Form("Gauss_cut_ds%i.pdf(",dsid));
  if(i==h_Gauss_cut.size()-1) can->Print(Form("Gauss_cut_ds%i.pdf)",dsid));
  if(i>0 && i<h_Gauss_cut.size()-1) can->Print(Form("Gauss_cut_ds%i.pdf",dsid));
  delete can;
 }
 
 TFile* outfile=new TFile("corr.root","RECREATE");
 for(int i=0;i<hname.size();i++)
  outfile->Add(h_corr[i]);
 outfile->Write();
 //outfile->Close();
 */
 
}

void check_gauss()
{
 
 int dsid=433859;
 
 TFile* file=TFile::Open(Form("output/ds%i.secondPCA.ver01.root",dsid));
 
 //throw random numbers accoridng to the mean and rms
 //send them through the inverse pca
 //overlay with the gauss (inputs to the pca)

 int npca=5;
 
 for(int bin=1;bin<=npca;bin++)
 {
   
   file->cd(Form("bin%i/pca",bin));
   
   IntArray* layer_array=(IntArray*)gDirectory->Get("RelevantLayers");
   
   TVectorD* Gauss_means   =(TVectorD*)gDirectory->Get("Gauss_means");
   TVectorD* Gauss_rms     =(TVectorD*)gDirectory->Get("Gauss_rms");
   
   TMatrixDSym* symCov     =(TMatrixDSym*)gDirectory->Get("symCov");
   TVectorD* MeanValues    =(TVectorD*)gDirectory->Get("MeanValues");
   TVectorD* SigmaValues   =(TVectorD*)gDirectory->Get("SigmaValues");
   
   TMatrixDSymEigen cov_eigen(*symCov);
   TMatrixD *EV = new TMatrixD(cov_eigen.GetEigenVectors());
   
   TVectorD* LowerBounds   =(TVectorD*)gDirectory->Get("LowerBounds");
   
   TRandom3* Random3=new TRandom3(); Random3->SetSeed(0);
   
   vector<int> RelevantLayers;
   for(int i=0;i<layer_array->GetSize();i++) RelevantLayers.push_back(layer_array->GetAt(i));
   
   vector<std::string> layer;
   vector<int> layerNr;
   for(unsigned int i=0;i<RelevantLayers.size();i++)
    layerNr.push_back(RelevantLayers[i]);
   for(unsigned int i=0;i<layerNr.size();i++)
   {
    string thislayer=Form("layer%i",layerNr[i]);
    layer.push_back(thislayer);
   }
   layer.push_back("totalE");
   
   vector<TH1D*> h_cumul;
   vector<TH1D*> h_sim_input;
   file->cd(Form("bin%i",bin));
   for(unsigned int l=0;l<layer.size();l++)
   {
    TH1D* hist=(TH1D*)gDirectory->Get(Form("h_cumul_%s",layer[l].c_str()));
    h_cumul.push_back(hist);
    TH1D* hist2=(TH1D*)gDirectory->Get(Form("h_input_%s",layer[l].c_str()));
    hist2->SetName(Form("h_sim_input_%s",layer[l].c_str()));
    h_sim_input.push_back(hist2);
   }
   
   TH1D* h_fractionsum=new TH1D("h_fractionsum","h_fractionsum",100,-1,3);
   vector<TH1D*> h_gauss;
   vector<TH1D*> h_gauss_input;
   vector<TH1D*> h_sim;
   for(unsigned int l=0;l<layer.size();l++)
   {
    TH1D* hist=new TH1D(Form("h_gauss_%s",layer[l].c_str()),Form("h_gauss_%s",layer[l].c_str()),200,-6,6);
    h_gauss.push_back(hist);
    TH1D* hist2=new TH1D(Form("h_gauss_input_%s",layer[l].c_str()),Form("h_gauss_input_%s",layer[l].c_str()),200,-6,6);
    h_gauss_input.push_back(hist2);
    TH1D* hist3=new TH1D(Form("h_sim_%s",layer[l].c_str()),Form("h_sim_%s",layer[l].c_str()),h_sim_input[l]->GetNbinsX(),
                         h_sim_input[l]->GetXaxis()->GetXmin(),h_sim_input[l]->GetXaxis()->GetXmax());
    h_sim.push_back(hist3);
   }
   
   double* vals_gauss_means=(double*)Gauss_means->GetMatrixArray();
   double* vals_gauss_rms  =Gauss_rms->GetMatrixArray();
   double* vals_lowerBounds=LowerBounds->GetMatrixArray();
   
   for(unsigned int l=0;l<layer.size();l++)
    cout<<"BIN "<<bin<<" "<<layer[l]<<" lowerBound "<<vals_lowerBounds[l]<<endl;
   
   int ntoys=10000;
   
   //loop over toys
   TRandom3 *ran=new TRandom3(0);
   for(int t=0;t<ntoys;t++)
   {
    
    double *output_data = new double[layer.size()] ;
    double *input_data = new double[layer.size()]  ;
    
    for(unsigned int l=0;l<layer.size();l++)
    {
     double mean=vals_gauss_means[l];
     double rms =vals_gauss_rms[l];
     double gauszz=Random3->Gaus(mean,rms);
     input_data[l]=gauszz;
    }
    
    P2X(SigmaValues, MeanValues, EV, layer.size(), input_data, output_data, layer.size());
    
    int do_rescale=1;
    
    double fractionsum=0.0;
    vector<double> sims;
    for(unsigned int l=0;l<layer.size();l++)
    {
     h_gauss[l]->Fill(output_data[l]);
     
     double uniform=(TMath::Erf(output_data[l]/1.414213562)+1)/2.f;
     //double uniform=ran->Uniform(0,1);
     double sim=get_inverse(uniform,h_cumul[l]);
     //cout<<"l "<<layer[l]<<" data "<<output_data[l]<<" uniform "<<uniform<<" sim "<<sim<<endl;
     if(l!=layer.size()-1) fractionsum+=sim;
     sims.push_back(sim);
    }
    h_fractionsum->Fill(fractionsum);
    double rescale=1.0;
    if(do_rescale) rescale=1.0/fractionsum;
    for(unsigned int l=0;l<layer.size();l++)
    {
     h_sim[l]->Fill(sims[l]*rescale);
    }
    sims.clear();
   
   } //for toys
   
   file->cd(Form("bin%i/",bin));
   TTree* T_Gauss=(TTree*)gDirectory->Get(Form("T_Gauss"));
   TreeReader* tree_Gauss = new TreeReader();
   tree_Gauss->SetTree(T_Gauss);
    cout<<"bin "<<bin<<" entries "<<tree_Gauss->GetEntries()<<endl;
   for(int event=0;event<tree_Gauss->GetEntries();event++)
   {
    tree_Gauss->GetEntry(event);
    for (unsigned int l=0;l<layer.size();l++)
    {
     double gauss= tree_Gauss->GetVariable(Form("energy_gauss_%s",layer[l].c_str()));
     h_gauss_input[l]->Fill(gauss);
    }
    
   }
   
   //make plots
   for(unsigned int l=0;l<layer.size();l++)
   {
    
    TCanvas* can=new TCanvas("can","can",0,0,800,600);
    h_gauss[l]->Draw("hist");
    h_gauss[l]->GetXaxis()->SetTitle(Form("Gauss for %s in PCA bin %i",layer[l].c_str(),bin));
    h_gauss_input[l]->Draw("histsame");
    h_gauss_input[l]->SetLineColor(2);
    h_gauss_input[l]->SetLineStyle(2);
    h_gauss_input[l]->Scale(h_gauss[l]->Integral()/h_gauss_input[l]->Integral());
    if(l==0) can->Print(Form("plots/pca_checks/ds%i_gauss_compare_bin%i.pdf(",dsid,bin));
    else if(l==layer.size()-1) can->Print(Form("plots/pca_checks/ds%i_gauss_compare_bin%i.pdf)",dsid,bin));
    else can->Print(Form("plots/pca_checks/ds%i_gauss_compare_bin%i.pdf",dsid,bin));
    delete can;
    
    TCanvas* can2=new TCanvas("can2","can2",0,0,1200,600);
    can2->Divide(2);
    can2->cd(1);
    h_sim_input[l]->Rebin(5);
    h_sim[l]->Rebin(5);
    h_sim_input[l]->SetLineWidth(2);
    h_sim[l]->SetLineWidth(1);
    h_sim_input[l]->Draw("hist");
    h_sim_input[l]->GetXaxis()->SetTitle(Form("Energy for %s in PCA bin %i",layer[l].c_str(),bin));
    h_sim[l]->Draw("histsame");
    h_sim[l]->SetLineColor(2);
    h_sim[l]->SetLineStyle(2);
    h_sim[l]->Scale(h_sim_input[l]->Integral()/h_sim[l]->Integral());
    can2->cd(2);
    TH1D* h_sim_input_log=(TH1D*)h_sim_input[l]->Clone("h_sim_input_log");
    TH1D* h_sim_log=(TH1D*)h_sim[l]->Clone("h_sim_log");
    h_sim_input_log->Draw("hist");
    can2->cd(2)->SetLogy();
    h_sim_log->Draw("histsame");
    if(l==0) can2->Print(Form("plots/pca_checks/ds%i_sim_bin%i.pdf(",dsid,bin));
    else if(l==layer.size()-1) can2->Print(Form("plots/pca_checks/ds%i_sim_bin%i.pdf)",dsid,bin));
    else can2->Print(Form("plots/pca_checks/ds%i_sim_bin%i.pdf",dsid,bin));
    delete can2;
    
   }
   
   TCanvas* can3=new TCanvas("can3","can3",0,0,800,600);
   h_fractionsum->Draw("hist");
   h_fractionsum->GetXaxis()->SetTitle(Form("Sum of fractions for bin %i",bin));
   can3->Print(Form("plots/pca_checks/ds%i_fractionsum_bin%i.pdf",dsid,bin));
   
  delete EV;
  
 } //for bin
 
}

void P2X(TVectorD* SigmaValues, TVectorD* MeanValues, TMatrixD *EV, int gNVariables, double *p, double *x, int nTest)
{

  double* gSigmaValues  = SigmaValues->GetMatrixArray();
  double* gMeanValues   = MeanValues->GetMatrixArray();
  double* gEigenVectors = EV->GetMatrixArray();

  for(int i = 0; i < gNVariables; i++)
    {
      x[i] = gMeanValues[i];
      for(int j = 0; j < nTest; j++)
        {
          x[i] += p[j] * gSigmaValues[i] * (double)(gEigenVectors[i *  gNVariables + j]);
        }
    }
}


double get_inverse(double rnd,TH1D* h_cumul)
{
  
  vector<float> m_HistoBorders;
  vector<float> m_HistoContents;
  
  for(int b=1;b<=h_cumul->GetNbinsX();b++)
   m_HistoBorders.push_back((float)h_cumul->GetBinLowEdge(b));
  m_HistoBorders.push_back((float)h_cumul->GetXaxis()->GetXmax());
  
  for(int b=1;b<h_cumul->GetNbinsX();b++)
   m_HistoContents.push_back(h_cumul->GetBinContent(b));
  m_HistoContents.push_back(1);
  
  double value = 0.;
  
  if(rnd<m_HistoContents[0])
  {
   double x1=m_HistoBorders[0];
   double x2=m_HistoBorders[1];
   double y1=0;
   double y2=m_HistoContents[0];
   double x=non_linear(y1,y2,x1,x2,rnd);
   value=x;
  }
  else
  {
   //find the first HistoContent element that is larger than rnd:
   vector<float>::iterator larger_element = std::upper_bound(m_HistoContents.begin(), m_HistoContents.end(), rnd);
   int index=larger_element-m_HistoContents.begin();
   double y=m_HistoContents[index];
   double x1=m_HistoBorders[index];
   double x2=m_HistoBorders[index+1];
   double y1=m_HistoContents[index-1];
   double y2=y;
   if((index+1)==((int)m_HistoContents.size()-1))
   {
    x2=m_HistoBorders[m_HistoBorders.size()-1];
    y2=m_HistoContents[m_HistoContents.size()-1];
   }
   double x=non_linear(y1,y2,x1,x2,rnd);
   value=x;
  }
  
  return value;
  
}


double non_linear(double y1,double y2,double x1,double x2,double y)
{
  double x=-1;
  double eps=0.0000000001;
  if((y2-y1)<eps)
  {
    x=x1;
  }
  else
  {
    double b=(x1-x2)/(sqrt(y1)-sqrt(y2));
    double a=x1-b*sqrt(y1);
    x=a+b*sqrt(y);
  }
  return x;
}

void highstat_test()
{
 
 TFile* file=TFile::Open("output/ds433859.secondPCA.ver01.root");
 
 TH1D* h_cumul=(TH1D*)file->Get("bin4/h_cumul_layer6");
 h_cumul->SetName("h_cumul");
 
 TH1D* h_original=(TH1D*)file->Get("bin4/h_input_layer6");
 h_original->SetName("h_original");
 
 TH1D* h_inverse=(TH1D*)h_original->Clone("h_inverse");
 h_inverse->Reset();
 
 vector<float> m_HistoBorders;
 vector<float> m_HistoContents;
  
 for(int b=1;b<=h_cumul->GetNbinsX();b++)
  m_HistoBorders.push_back((float)h_cumul->GetBinLowEdge(b));
 m_HistoBorders.push_back((float)h_cumul->GetXaxis()->GetXmax());
  
 for(int b=1;b<h_original->GetNbinsX();b++)
  m_HistoContents.push_back(h_cumul->GetBinContent(b));
 m_HistoContents.push_back(1);
 
 int ntoys=100000;
 
 TRandom3* ran=new TRandom3(0);
 
 for(int t=0;t<ntoys;t++)
 {
  
  double rnd=ran->Uniform(0,1);
  
  double value = 0.;
  
  if(rnd<m_HistoContents[0])
  {
   double x1=m_HistoBorders[0];
   double x2=m_HistoBorders[1];
   double y1=0;
   double y2=m_HistoContents[0];
   double x=non_linear(y1,y2,x1,x2,rnd);
   value=x;
  }
  else
  {
   //find the first HistoContent element that is larger than rnd:
   vector<float>::iterator larger_element = std::upper_bound(m_HistoContents.begin(), m_HistoContents.end(), rnd);
   int index=larger_element-m_HistoContents.begin();
   double y=m_HistoContents[index];
   double x1=m_HistoBorders[index];
   double x2=m_HistoBorders[index+1];
   double y1=m_HistoContents[index-1];
   double y2=y;
   if((index+1)==((int)m_HistoContents.size()-1))
   {
    x2=m_HistoBorders[m_HistoBorders.size()-1];
    y2=m_HistoContents[m_HistoContents.size()-1];
   }
   double x=non_linear(y1,y2,x1,x2,rnd);
   value=x;
  }
  
  h_inverse->Fill(value);
  
 } //for ntoys
 
 TCanvas* can=new TCanvas("can","can",0,0,800,600);
 h_original->Draw("hist");
 h_inverse->Scale(h_original->Integral()/h_inverse->Integral());
 h_inverse->SetLineColor(2); h_inverse->SetLineStyle(2);
 h_inverse->Draw("histsame");
 
 
}