using namespace std;

#include "TMatrixD.h"
#include "TVectorD.h"
#include "TH1D.h"
#include "TFile.h"
#include "TROOT.h"
#include "TSystem.h"
#include "TApplication.h"
#include "TTree.h"
#include "TSystem.h"
#include "TH2D.h"
#include "TPrincipal.h"
#include "TMath.h"
#include "TBrowser.h"
#include "firstPCA.h"
#include "TreeReader.h"
#include "TLorentzVector.h"
#include "TChain.h"

#include <iostream>

firstPCA::firstPCA()
{
  //default parameters:
  m_numberfinebins=5000;
  m_edepositcut=0.001;
  m_cut_eta_low=-100;
  m_cut_eta_high=100;
  m_nbins1=5;
  m_nbins2=1;
  m_debuglevel=0;
  m_apply_etacut=1;
  
  m_outfilename="";
  m_chain = 0;
}

firstPCA::firstPCA(TChain* chain,string outfilename)
{
  //default parameters:
  m_numberfinebins=5000;
  m_edepositcut=0.001;
  m_cut_eta_low=-100;
  m_cut_eta_high=100;
  m_nbins1=5;
  m_nbins2=1;
  m_debuglevel=0;

  m_outfilename=outfilename;
  m_chain=chain;

}

void firstPCA::apply_etacut(int flag)
{
 m_apply_etacut=flag;
}

void firstPCA::set_cumulativehistobins(int bins)
{
  m_numberfinebins=bins;
}

void firstPCA::set_edepositcut(double cut)
{
  m_edepositcut=cut;
}

void firstPCA::set_etacut(double cut_low, double cut_high)
{
  m_cut_eta_low=cut_low;
  m_cut_eta_high=cut_high;
}

void firstPCA::set_pcabinning(int bin1,int bin2)
{
  m_nbins1=bin1;
  m_nbins2=bin2;
}

void firstPCA::run()
{
  cout<<endl;
  cout<<"****************"<<endl;
  cout<<"     1st PCA"<<endl;
  cout<<"****************"<<endl;
  cout<<endl;
  cout<<"Now running firstPCA with the following parameters:"<<endl;
  cout<<"  Energy deposit cut: "<<m_edepositcut<<endl;
  cout<<"  PCA binning: 1st compo "<<m_nbins1<<", 2nd compo "<<m_nbins2<<endl;
  cout<<"  Number of bins in the cumulative histograms "<<m_numberfinebins<<endl;
  cout<<"  Eta cut: "<<m_cut_eta_low<<" "<<m_cut_eta_high<<endl;
  cout<<"  Apply eta cut: "<<m_apply_etacut<<endl;
  cout<<endl;
  if(m_debuglevel) cout<<"initialize input tree reader"<<endl;
  TreeReader* read_inputTree = new TreeReader();
  read_inputTree->SetTree(m_chain);

  vector<int> layerNr;
  cout<<"--- Get relevant layers and Input Histograms"<<endl;
  vector<TH1D*> histos_data=get_relevantlayers_inputs(layerNr,read_inputTree);

  vector<string> layer;
  for(unsigned int l=0;l<layerNr.size();l++)
    layer.push_back(Form("layer%i",layerNr[l]));
  layer.push_back("totalE");

  vector<TH1D*> cumul_data =get_cumul_histos(layer, histos_data);

  cout<<"--- Now define the TPrincipal"<<endl;
  TPrincipal* principal = new TPrincipal(layer.size(),"ND");  //ND means normalize cov matrix and store data

  TTree *T_Gauss = new TTree("T_Gauss","T_Gauss");
  T_Gauss->SetDirectory(0);
  double* data_Gauss = new double[layer.size()];
  int eventNumber;
  for(unsigned int l=0;l<layer.size();l++)
  {
    T_Gauss->Branch(Form("data_Gauss_%s",layer[l].c_str()),&data_Gauss[l],Form("data_Gauss_%s/D",layer[l].c_str()));
    T_Gauss->Branch("eventNumber",&eventNumber,"eventNumber/I");
  }

  cout<<"--- Uniformization/Gaussianization"<<endl;
  for(int event=0;event<read_inputTree->GetEntries();event++)
  {
    read_inputTree->GetEntry(event);
    eventNumber=event;

    double E  = read_inputTree->GetVariable("TruthE");
    double px = read_inputTree->GetVariable("TruthPx");
    double py = read_inputTree->GetVariable("TruthPy");
    double pz = read_inputTree->GetVariable("TruthPz");

    TLorentzVector tlv;  tlv.SetPxPyPzE(px,py,pz,E);
    bool pass_eta=0;
    if(!m_apply_etacut) pass_eta=1;
    if(m_apply_etacut)  pass_eta=(fabs(tlv.Eta())>m_cut_eta_low && fabs(tlv.Eta())<m_cut_eta_high);
    if(pass_eta)
    {
      double total_e=read_inputTree->GetVariable("total_cell_energy");
      for(unsigned int l=0;l<layer.size();l++)
      {
        double data = 0.;
        //double data_uniform;

        if(l==layer.size()-1)
          data = total_e;
        else
          data = read_inputTree->GetVariable(Form("cell_energy[%d]",layerNr[l]))/total_e;

        //Uniformization
        double cumulant = get_cumulant(data,cumul_data[l]);
        cumulant = TMath::Min(cumulant,1.-10e-10);
        cumulant = TMath::Max(cumulant,0.+10e-10);
        //data_uniform = cumulant;

        //Gaussianization
        double maxErfInvArgRange = 0.99999999;
        double arg = 2.0*cumulant - 1.0;
        arg = TMath::Min(+maxErfInvArgRange,arg);
        arg = TMath::Max(-maxErfInvArgRange,arg);
        data_Gauss[l] = 1.414213562*TMath::ErfInverse(arg);

      } //for layers

      //Add this datapoint to the PCA
      principal->AddRow(data_Gauss);
      T_Gauss->Fill();

    } //pass eta
    if(event%2000==0) cout<<event<<" from "<<read_inputTree->GetEntries()<<" done "<<endl;
  } //event loop
  
  cout<<"--- MakePrincipals()"<<endl;
  principal->MakePrincipals();

  cout<<"--- PCA Results"<<endl;
  principal->Print("MSE");

  cout<<"--- PCA application and binning of first principal component"<<endl;

  TreeReader* tree_Gauss = new TreeReader();
  tree_Gauss->SetTree(T_Gauss);

  int Bin_1stPC1=0;
  int Bin_1stPC2=0;

  TH1D* hPCA_first_component = new TH1D("hPCA_first_component","hPCA_first_component",10000,-20,20);

  for(int event=0;event<tree_Gauss->GetEntries();event++)
  {
    tree_Gauss->GetEntry(event);

    double* data_PCA = new double[layer.size()];
    double* input_data = new double[layer.size()];

    for (unsigned int l=0;l<layer.size(); l++)
      input_data[l] = tree_Gauss->GetVariable(Form("data_Gauss_%s",layer[l].c_str()));

    principal->X2P(input_data,data_PCA);  
    hPCA_first_component->Fill(data_PCA[0]); 
  }

  cout<<"--- Binning 1st Principal Component" <<endl;
  double *xq = new double[m_nbins1];
  double *yq = new double[m_nbins1];

  //2D binning:
  quantiles( hPCA_first_component, m_nbins1, xq , yq );
  for (int m = 0; m < m_nbins1 ; m++)
  {
    int a=1;
    if(m>0) a=hPCA_first_component->FindBin(yq[m-1]);
    int b=hPCA_first_component->FindBin(yq[m]);
    cout<<"Quantiles "<<m+1<<"  "<<xq[m]<<" "<<yq[m]<<" -> Events "<<hPCA_first_component->Integral(a,b-1)<<endl;
  }

  std::vector<TH1D*> h_compo1;
  for(int m=0;m<m_nbins1;m++)
    h_compo1.push_back(new TH1D(Form("h_compo1_bin%i",m),Form("h_compo1_bin%i",m),20000,-20,20));

  cout<<"--- PCA application and binning 2nd principal component"<<endl;
  for(int event=0;event<tree_Gauss->GetEntries();event++)
  {
    tree_Gauss->GetEntry(event);
    double* data_PCA = new double[layer.size()];
    double* input_data = new double[layer.size()];

    for (unsigned int l=0;l<layer.size();l++)
      input_data[l] = tree_Gauss->GetVariable(Form("data_Gauss_%s",layer[l].c_str()));

    principal->X2P(input_data,data_PCA);

    int firstbin=-42;
    //Binning 1st PC
    for(int m = 0 ; m < m_nbins1 ; m++)
    {
      if(m==0 && data_PCA[0] <= yq[0])
        firstbin = 0;
      if(m > 0 && data_PCA[0] > yq[m-1] && data_PCA[0] <= yq[m])
        firstbin = m;
    }

    if (firstbin >= 0) h_compo1[firstbin]->Fill(data_PCA[1]);

  }

  // non-standard
  //double yq2d[m_nbins1][m_nbins2];
  std::vector<std::vector<double> > yq2d (m_nbins1, std::vector<double>(m_nbins2));

  for(int m=0;m<m_nbins1;m++)
  {
    if(m_debuglevel) cout<<"now do m "<<m<<endl;
    double *xq2 = new double[m_nbins2];
    double *yq2 = new double[m_nbins2];
    quantiles( h_compo1[m], m_nbins2, xq2 , yq2);
    if(m_debuglevel) {
      cout<<"1stPCA bin# "<<m<<" Events "<<h_compo1[m]->Integral()<<endl;
    }

    for (int u = 0; u < m_nbins2 ; u++)
    {
      int a=0;
      if(u>0) a=h_compo1[m]->FindBin(yq2[u-1]);
      int b=h_compo1[m]->FindBin(yq2[u]);
      cout<<"Quantile # "<<u<<"  "<<xq2[u]<<" "<<yq2[u]<<" -> Events "<<h_compo1[m]->Integral(a,b-1)<<endl;
    }

    for(int u=0;u<m_nbins2;u++)
      yq2d[m][u]=yq2[u];

    delete [] xq2;
    delete [] yq2;
  }

  // cleanup
  for (auto it = h_compo1.begin(); it != h_compo1.end(); ++it)
    delete *it;
  h_compo1.clear();

  cout<<"--- Fill a tree that has the bin information"<<endl;
  int firstPCAbin;
  double* data = new double[layer.size()];
  TTree* tree_1stPCA=new TTree(Form("tree_1stPCA"),Form("tree_1stPCA"));
  tree_1stPCA->SetDirectory(0);
  tree_1stPCA->Branch("firstPCAbin",&firstPCAbin,"firstPCAbin/I");
  for(unsigned int l=0;l<layer.size();l++)
    tree_1stPCA->Branch(Form("energy_%s",layer[l].c_str()),&data[l],Form("energy_%s/D",layer[l].c_str()));

  for(int event=0;event<tree_Gauss->GetEntries();event++)
  {
    tree_Gauss->GetEntry(event);
    double* data_PCA = new double[layer.size()];
    double* input_data = new double[layer.size()];
    int eventNumber=tree_Gauss->GetVariable("eventNumber");

    for(unsigned int l=0; l<layer.size();l++)
      input_data[l] = tree_Gauss->GetVariable(Form("data_Gauss_%s",layer[l].c_str()));

    //PCA Application
    principal->X2P(input_data,data_PCA);

    //Binning 1st and 2nd PC
    for(int m = 0 ; m < m_nbins1 ; m++)
    {
      if(m==0 && data_PCA[0]<=yq[m])
      {
        Bin_1stPC1 = 0;
        for(int u=0;u<m_nbins2;u++)
        {
          if(u==0 && data_PCA[1]<=yq2d[0][u])
            Bin_1stPC2 = 0;
          if(u>0 && data_PCA[1]>yq2d[0][u-1] && data_PCA[1]<=yq2d[0][u])
            Bin_1stPC2 = u;
        }
      }
      if(m>0 && data_PCA[0]>yq[m-1] && data_PCA[0]<=yq[m])
      {
        Bin_1stPC1 = m;
        for(int u=0;u<m_nbins2;u++)
        {
          if(u==0 && data_PCA[1]<=yq2d[m][u])
            Bin_1stPC2 = 0;
          if(u>0 && data_PCA[1]>yq2d[m][u-1] && data_PCA[1]<=yq2d[m][u])
            Bin_1stPC2 = u;
        }
      }   
    }

    firstPCAbin=Bin_1stPC1+m_nbins1*Bin_1stPC2+1;

    //find the energy fractions and total energy for that given event
    read_inputTree->GetEntry(eventNumber);
    for(unsigned int l=0;l<layer.size();l++)
    {
      if(l==layer.size()-1)
        data[l] = read_inputTree->GetVariable("total_cell_energy");
      else
        data[l] = read_inputTree->GetVariable(Form("cell_energy[%d]",layerNr[l]))/read_inputTree->GetVariable("total_cell_energy");
    }

    tree_1stPCA->Fill();

  } //for events in gauss

  //add a histogram that holds the relevant layer:

  int totalbins=m_nbins1*m_nbins2;

  TH2I* h_layer=new TH2I("h_layer","h_layer",totalbins,0.5,totalbins+0.5,25,-0.5,24.5);
  h_layer->GetXaxis()->SetTitle("PCA bin");
  h_layer->GetYaxis()->SetTitle("Layer");
  for(int b=0;b<totalbins;b++)
  {
    for(int l=0;l<25;l++)
    {
      int is_relevant=0;
      for(unsigned int i=0;i<layerNr.size();i++)
      {
        if(l==layerNr[i]) is_relevant=1;
      }
      h_layer->SetBinContent(b+1,l+1,is_relevant);
    }
  }

  TFile* output=TFile::Open(m_outfilename.c_str(),"RECREATE");
  output->Add(h_layer);
  output->Add(tree_1stPCA);
  output->Write();
  
  cout<<"1st PCA is done. Output file: "<<m_outfilename<<endl;
  
  // cleanup
  delete read_inputTree;
  delete tree_Gauss;
  delete principal;
  delete T_Gauss;

  delete [] xq;
  delete [] yq;

} // run

vector<TH1D*> firstPCA::get_cumul_histos(vector<string> layer, vector<TH1D*> histos)
{

  vector<TH1D*> cumul;

  for(unsigned int i=0;i<histos.size();i++)
  {
    TH1D* h_cumul=(TH1D*)histos[i]->Clone(Form("h_cumul_%s",layer[i].c_str()));
    for (int b=1; b<=h_cumul->GetNbinsX(); b++)
    {
      h_cumul->SetBinContent(b,histos[i]->Integral(1,b));
      h_cumul->SetBinError(b,0);
    }
    cumul.push_back(h_cumul);
  }

  return cumul;

}

vector<TH1D*> firstPCA::get_relevantlayers_inputs(vector<int> &layerNr, TreeReader* read_inputTree)
{

  int NLAYER=25;

  vector<TH1D*>  data;
  vector<double> MaxInputs;
  vector<double> MinInputs;

  double max_e=0.0;
  double min_e=100000000;

  vector<double> sum_efraction;
  for(int l=0;l<NLAYER;l++)
  {
    sum_efraction.push_back(0.0);
    MaxInputs.push_back(0.0);
    MinInputs.push_back(1000000.0);
  }

  int N_pass_eta=0;
  for(int event=0;event<read_inputTree->GetEntries();event++ )
  {
    read_inputTree->GetEntry(event);
    double E  = read_inputTree->GetVariable("TruthE");
    double px = read_inputTree->GetVariable("TruthPx");
    double py = read_inputTree->GetVariable("TruthPy");
    double pz = read_inputTree->GetVariable("TruthPz");
    TLorentzVector tlv; tlv.SetPxPyPzE(px,py,pz,E);
    bool pass_eta=0;
    if(!m_apply_etacut) pass_eta=1;
    if(m_apply_etacut)  pass_eta=(fabs(tlv.Eta())>m_cut_eta_low && fabs(tlv.Eta())<m_cut_eta_high);
    if(pass_eta)
    {
      N_pass_eta++;
      double total_e=read_inputTree->GetVariable("total_cell_energy");
      for(int l=0;l<NLAYER;l++)
      {
        double efraction = read_inputTree->GetVariable(Form("cell_energy[%d]",l))/total_e;
        sum_efraction[l] += efraction;
        if(efraction>MaxInputs[l]) MaxInputs[l]=efraction;
        if(efraction<MinInputs[l]) MinInputs[l]=efraction;
        if(total_e>max_e) max_e=total_e;
        if(total_e<min_e) min_e=total_e;
      }
    }
    if(event%2000==0) cout<<event<<" from "<<read_inputTree->GetEntries()<<" done"<<endl;
  }

  cout<<"rel. layer"<<endl;

  for(int l=0;l<NLAYER;l++)
  {
    if(N_pass_eta>0)
    {
      if(sum_efraction[l]/N_pass_eta>=m_edepositcut)
      {
        layerNr.push_back(l);
        cout<<"Layer "  <<l <<" is relevant! sum_efraction= "<<sum_efraction[l]<<" sum/entries= "<<sum_efraction[l]/N_pass_eta<<endl;
      }
    }
  }

  for(unsigned int k=0;k<layerNr.size();k++)
    cout<<"Relevant "<<layerNr[k]<<endl;

  cout<<"init data histos"<<endl;

  for(int l=0;l<NLAYER;l++)
  {
    int is_rel=0;
    for(unsigned int k=0;k<layerNr.size();k++)
    {
      if(l==layerNr[k])
        is_rel=1;
    }
    if(is_rel)
    {
      TH1D* h_data=new TH1D(Form("h_data_layer%i",l),Form("h_data_layer%i",l),m_numberfinebins,MinInputs[l],MaxInputs[l]);
      h_data->Sumw2();
      data.push_back(h_data);
    }
  }
  TH1D* h_data_total=new TH1D("h_data_totalE","h_data_totalE",m_numberfinebins,min_e,max_e);
  data.push_back(h_data_total);

  cout<<"fill data histos"<<endl;

  for(int event=0;event<read_inputTree->GetEntries();event++)
  {
    read_inputTree->GetEntry(event);
    TLorentzVector tlv;  tlv.SetPxPyPzE(read_inputTree->GetVariable("TruthPx"),read_inputTree->GetVariable("TruthPy"),read_inputTree->GetVariable("TruthPz"),read_inputTree->GetVariable("TruthE"));
    bool pass_eta=0;
    if(!m_apply_etacut) pass_eta=1;
    if(m_apply_etacut)  pass_eta=(fabs(tlv.Eta())>m_cut_eta_low && fabs(tlv.Eta())<m_cut_eta_high);
    if(pass_eta)
    {
      double total_e=read_inputTree->GetVariable("total_cell_energy");
      ((TH1D*)data[data.size()-1])->Fill(total_e);
      for(unsigned int l=0;l<layerNr.size();l++)
      {
        ((TH1D*)data[l])->Fill(read_inputTree->GetVariable(Form("cell_energy[%d]",layerNr[l]))/total_e);
      }
    }
    if(event%2000==0) cout<<event<<" from "<<read_inputTree->GetEntries()<<" done"<<endl;
  }
  
  for(unsigned int l=0;l<data.size();l++)
  {
    ((TH1D*)data[l])->Scale(1.0/data[l]->Integral());
  }
  
  return data;
 
}


double firstPCA::get_cumulant(double x, TH1D* h)
{

  //Cumulant "à la TMVA"

  int nbin = h->GetNbinsX();
  int bin = h->FindBin(x);
  bin = TMath::Max(bin,1);
  bin = TMath::Min(bin,h->GetNbinsX());


  double AvNuEvPerBin;
  double Tampon = 0 ;
  for (int i=1; i<=nbin; i++) {
    Tampon += h->GetBinContent(i);
  }

  AvNuEvPerBin = Tampon/nbin;

  double cumulant;
  double x0, x1, y0, y1;
  double total = h->GetNbinsX()*AvNuEvPerBin;
  double supmin = 0.5/total;

  x0 = h->GetBinLowEdge(TMath::Max(bin,1));
  x1 = h->GetBinLowEdge(TMath::Min(bin,h->GetNbinsX())+1);

  y0 = h->GetBinContent(TMath::Max(bin-1,0)); // Y0 = F(x0); Y0 >= 0
  y1 = h->GetBinContent(TMath::Min(bin, h->GetNbinsX()+1));  // Y1 = F(x1);  Y1 <= 1

  //Zero bin
  if (bin == 0) {
    y0 = supmin;
    y1 = supmin;
  }
  if (bin == 1) {
    y0 = supmin;
  }
  if (bin > h->GetNbinsX()) {
    y0 = 1.-supmin;
    y1 = 1.-supmin;
  }
  if (bin == h->GetNbinsX()) {
    y1 = 1.-supmin;
  }

  ////////////////////////

  if (x0 == x1) {
    cumulant = y1;
  } else {
    cumulant = y0 + (y1-y0)*(x-x0)/(x1-x0);
  }

  if (x <= h->GetBinLowEdge(1)){
    cumulant = supmin;
  }
  if (x >= h->GetBinLowEdge(h->GetNbinsX()+1)){
    cumulant = 1-supmin;
  }

  return cumulant;

}


void firstPCA::quantiles(TH1D* h, int nq, double* xq, double* yq)
{

  //Function for quantiles
  // h Input histo
  // nq number of quantiles
  // xq position where to compute the quantiles in [0,1]
  // yq array to contain the quantiles

  for (int i=0;i<nq;i++)
  {
    xq[i] = double(i+1)/nq;
    h->GetQuantiles(nq,yq,xq);
  }

}
