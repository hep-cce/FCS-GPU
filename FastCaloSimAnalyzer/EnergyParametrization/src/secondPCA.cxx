/*
  Copyright (C) 2002-2021 CERN for the benefit of the ATLAS collaboration
*/

#include "TMatrixF.h"
#include "TMatrixD.h"
#include "TMatrixDSym.h"
#include "TMatrixDSymEigen.h"
#include "TVectorF.h"
#include "TRandom3.h"
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
#include "secondPCA.h"
#include "firstPCA.h"
#include "TFCS1DFunctionFactory.h"
#include "ISF_FastCaloSimEvent/TFCS1DFunction.h"
#include "TreeReader.h"
#include "ISF_FastCaloSimEvent/IntArray.h"
#include <CLHEP/Random/RandFlat.h>

#include <iostream>
#include <sstream>

using namespace std;

secondPCA::secondPCA(string firstpcafilename, string outfilename)
{
  m_firstpcafilename = firstpcafilename;
  m_outfilename      = outfilename;

  m_numberfinebins    = 5000;
  m_storeDetails      = 0;
  m_PCAbin            = -1; //-1 means all bins
  m_skip_regression   = 0;
  m_neurons_start     = 2;
  m_neurons_end       = 8;
  m_ntoys             = 1000;
  m_maxdev_regression = 5;
  m_maxdev_smartrebin = 5;
}

void secondPCA::set_cut_maxdeviation_regression(double val)
{
 m_maxdev_regression=val;
}

void secondPCA::set_cut_maxdeviation_smartrebin(double val)
{
 m_maxdev_smartrebin=val;
}

void secondPCA::set_Ntoys(int val)
{
 m_ntoys=val;
}

void secondPCA::set_neurons_iteration(int start,int end)
{
  m_neurons_start = start;
  m_neurons_end   = end;
}

void secondPCA::set_storeDetails(int flag)
{
 m_storeDetails=flag;
}

void secondPCA::set_cumulativehistobins(int bins)
{
 m_numberfinebins=bins;
}

void secondPCA::set_PCAbin(int bin)
{
 m_PCAbin=bin;
}

void secondPCA::set_skip_regression(int flag)
{
 m_skip_regression=flag;
}

// void secondPCA::run()
void secondPCA::run(CLHEP::HepRandomEngine *randEngine)
{

  // Open inputfile:
  TFile* inputfile = TFile::Open( m_firstpcafilename.c_str(), "READ" );
  if ( !inputfile ) {
    cout << "ERROR: Inputfile could not be opened" << endl;
    // I don't think we can continue...
    return;
  }

 int nbins; int nbins0=1;
  vector<int> layerNr = getLayerBins( inputfile, nbins );

  /*
  //if a specific PCA bin was set,check if this is available, and set nbins to that
  if(m_PCAbin>0)
  {
         if(m_PCAbin<=nbins)
         {
          nbins =m_PCAbin;
          nbins0=m_PCAbin;
          string binlabel=Form("_bin%i",m_PCAbin);
    m_outfilename=m_outfilename+binlabel;
         }
         else cout<<"ERROR: The PCA bin you set is not available"<<endl;
  }
  */

  vector<string> layer;
 for(unsigned int l=0;l<layerNr.size();l++)
  layer.push_back(Form("layer%i",layerNr[l]));
  layer.push_back( "totalE" );

  int* samplings = new int[layerNr.size()];
 for(unsigned int i=0;i<layerNr.size();i++)
  samplings[i]=layerNr[i];

  cout << endl;
  cout << "****************" << endl;
  cout << "     2nd PCA" << endl;
  cout << "****************" << endl;
  cout << endl;
  cout << "Now running 2nd PCA with the following parameters:" << endl;
  cout << "   Input file (1st PCA): " << m_firstpcafilename << endl;
  cout << "   Number of bins of the cumulative histos: " << m_numberfinebins << endl;
  cout << "   storeDetails: " << m_storeDetails << endl;
  cout << "   PCA bin number(s): ";
  for ( int b = nbins0; b <= nbins; b++ ) cout << b << " ";
  cout << endl;
  cout << "   skip regression: " << m_skip_regression << endl;
  cout << "   Regression test toys:" << m_ntoys << endl;
  cout << "   Neurons in the regression iteration:" << m_neurons_start << " - " << m_neurons_end << endl;
  cout << "   Maximal deviation of approximated histogram (regression):  " << m_maxdev_regression << "%" << endl;
  cout << "   Maximal deviation of approximated histogram (smart-rebin): " << m_maxdev_smartrebin << "%" << endl;
  cout << endl;
  cout << "--- Init the TreeReader" << endl;
  TTree*      InputTree      = (TTree*)inputfile->Get( "tree_1stPCA" );
  TreeReader* read_inputTree = new TreeReader();
  read_inputTree->SetTree( InputTree );

  TFile* output = new TFile( m_outfilename.c_str(), "RECREATE" );
 for(int b=nbins0;b<=nbins;b++)
 {
    output->mkdir( Form( "bin%i", b ) );
    output->mkdir( Form( "bin%i/pca", b ) );
	for(unsigned int l=0;l<layer.size();l++)
   output->mkdir(Form("bin%i/%s",b,layer[l].c_str()));
  }

  // add the pca bin probability
  float* prob  = new float[nbins + 1];
  TH1I*  h_bin = new TH1I( "h_bin", "h_bin", nbins + 1, -0.5, nbins + 0.5 );
 for(int event=0;event<read_inputTree->GetEntries();event++)
 {
    read_inputTree->GetEntry( event );
    int bin = read_inputTree->GetVariable( "firstPCAbin" );
    h_bin->Fill( bin );
  }
 for(int i=0;i<=nbins;i++)
  prob[i]=(float)h_bin->GetBinContent(i+1)/(float)h_bin->Integral();
  TVectorF* PCAbinprob = new TVectorF( nbins + 1, prob );
  delete h_bin;

  PCAbinprob->Write( "PCAbinprob" );
  output->Write();
  output->Close();

 for(int b=nbins0;b<=nbins;b++)
 {
    cout << "--- now performing 2nd PCA in bin " << b << endl;
    // do_pca(layer, b, read_inputTree, samplings);
    do_pca( randEngine, layer, b, read_inputTree, samplings );
  }

  cout << "2nd PCA is done. Output: " << m_outfilename << endl;

  // cleanup
  delete read_inputTree;
  delete[] samplings;
}

// void secondPCA::do_pca(vector<string> layer, int bin, TreeReader* read_inputTree, int* samplings)
void secondPCA::do_pca(CLHEP::HepRandomEngine *randEngine, vector<string> layer, int bin, TreeReader* read_inputTree, int* samplings)
{

  cout << "check1 in do_pca for bin " << bin << endl;

  // make a tree that holds only the events for that
  TTree*  bintree = new TTree( "bintree", "bintree" );
  double* data    = new double[layer.size()];
  for ( unsigned int l = 0; l < layer.size(); l++ )
    bintree->Branch( Form( "energy_%s", layer[l].c_str() ), &data[l], Form( "energy_%s/D", layer[l].c_str() ) );
 for(int event=0;event<read_inputTree->GetEntries();event++)
 {
    read_inputTree->GetEntry( event );
    int firstPCAbin = read_inputTree->GetVariable( "firstPCAbin" );
  if(firstPCAbin==bin)
  {
   for(unsigned int l=0;l<layer.size();l++)
   {
        data[l] = read_inputTree->GetVariable( Form( "energy_%s", layer[l].c_str() ) );

        // cout<<"bin "<<bin<<" l "<<l<<" data[l] "<<data[l]<<endl;
      }
      bintree->Fill();
    }
  }

  cout << "check2 " << endl;

  // initialize the reader for this bintree
  TreeReader* read_bintree = new TreeReader();
  read_bintree->SetTree( bintree );

  vector<TH1D*> histos_data = get_histos_data( layer, read_bintree );
  vector<TH1D*> cumul_data  = get_cumul_histos( layer, histos_data );

  TPrincipal* principal = new TPrincipal( layer.size(), "ND" ); // ND means normalize cov matrix and store data
  TTree*      T_Gauss   = new TTree( "T_Gauss", "T_Gauss" );
  T_Gauss->SetDirectory( 0 );
  double* data_Gauss = new double[layer.size()];
  double* data_PCA   = new double[layer.size()];
 for(unsigned int l=0;l<layer.size();l++)
 {
  T_Gauss->Branch(Form("energy_gauss_%s",layer[l].c_str()),&data_Gauss[l],Form("energy_gauss_%s/D",layer[l].c_str()));
    T_Gauss->Branch( Form( "energy_pca_comp%i", l ), &data_PCA[l], Form( "energy_pca_%i/D", l ) );
  }

  TTree* T_Gauss0 = new TTree( "T_Gauss0", "T_Gauss0" );
  T_Gauss0->SetDirectory( 0 );
  double* data_Gauss0 = new double[layer.size()];
  for ( unsigned int l = 0; l < layer.size(); l++ )
  T_Gauss0->Branch(Form("energy_gauss0_%s",layer[l].c_str()),&data_Gauss0[l],Form("energy_gauss0_%s/D",layer[l].c_str()));

  cout << "check3 " << endl;

 for(int event=0;event<read_bintree->GetEntries();event++)
 {
    read_bintree->GetEntry( event );
  for(unsigned int l=0;l<layer.size();l++)
  {
      double data = read_bintree->GetVariable( Form( "energy_%s", layer[l].c_str() ) );
      // cout<<"l "<<l<<" "<<layer[l]<<" data "<<data<<endl;

      // double cumulant = get_cumulant_random(data,cumul_data[l]);
      double cumulant = get_cumulant_random( randEngine, data, cumul_data[l] );

      // Gaussianization
      double maxErfInvArgRange = 0.99999999;
      double arg               = 2.0 * cumulant - 1.0;
      arg                      = TMath::Min( +maxErfInvArgRange, arg );
      arg                      = TMath::Max( -maxErfInvArgRange, arg );
      data_Gauss0[l]           = TMath::Pi() / 2.0 * TMath::ErfInverse( arg );
      // cout<<"event "<<event<<" l "<<l<<" data_Gauss0 "<<data_Gauss0[l]<<endl;
    }
    principal->AddRow( data_Gauss0 );
    T_Gauss0->Fill();
  } // event loop

  cout << "check4 " << endl;
  principal->Print( "MSE" );

  principal->MakePrincipals();

  cout << std::endl << "- Principal Component Analysis Results in bin " << bin << endl;
  principal->Print( "MSE" );

  TreeReader* reader_treeGauss0 = new TreeReader();
  reader_treeGauss0->SetTree( T_Gauss0 );

  // second loop to fill the tree with the Gauss and PCA data
 for(int event=0;event<reader_treeGauss0->GetEntries();event++)
 {
    reader_treeGauss0->GetEntry( event );
  for(unsigned int l=0;l<layer.size();l++)
  {
      double data   = reader_treeGauss0->GetVariable( Form( "energy_gauss0_%s", layer[l].c_str() ) );
      data_Gauss[l] = data;
    }
    principal->X2P( data_Gauss, data_PCA );
    T_Gauss->Fill();
  } // event loop

  cout << "--- Application to get Mean and RMS of the PCA transformed data" << endl;
  TreeReader* reader_treeGauss = new TreeReader();
  reader_treeGauss->SetTree( T_Gauss );

 vector<double> data_PCA_min; vector<double> data_PCA_max;
 for(unsigned int l=0;l<layer.size();l++)
 {
    data_PCA_min.push_back( 100000.0 );
    data_PCA_max.push_back( -100000.0 );
  }

  cout << "check6" << endl;

 for(int event=0;event<reader_treeGauss->GetEntries();event++)
 { 
    reader_treeGauss->GetEntry( event );
    double* input_data = new double[layer.size()];
    double* data_PCA   = new double[layer.size()];

    for ( unsigned int l = 0; l < layer.size(); l++ )
      input_data[l] = reader_treeGauss->GetVariable( Form( "energy_gauss_%s", layer[l].c_str() ) );
    principal->X2P( input_data, data_PCA );
  for(unsigned int l=0;l<layer.size();l++)
  {
      if ( data_PCA[l] > data_PCA_max[l] ) data_PCA_max[l] = data_PCA[l];
      if ( data_PCA[l] < data_PCA_min[l] ) data_PCA_min[l] = data_PCA[l];
    }

    delete[] input_data;
    delete[] data_PCA;
  }

  cout << "check7" << endl;

  // fill histograms
  std::vector<TH1D*> h_data_PCA;
 for(unsigned int l=0;l<layer.size();l++)
 {
 	h_data_PCA.push_back(new TH1D(Form("h_data_PCA_%s",layer[l].c_str()),Form("h_data_PCA_%s",layer[l].c_str()),1000,data_PCA_min[l],data_PCA_max[l]));
  }

  cout << "check8" << endl;

 
 for(int event=0;event<reader_treeGauss->GetEntries();event++)
 {
    reader_treeGauss->GetEntry( event );
    double* input_data = new double[layer.size()];
    double* data_PCA   = new double[layer.size()];

    for ( unsigned int l = 0; l < layer.size(); l++ )
      input_data[l] = reader_treeGauss->GetVariable( Form( "energy_gauss_%s", layer[l].c_str() ) );
    principal->X2P( input_data, data_PCA );
  for(unsigned int l=0;l<layer.size();l++)
 	 h_data_PCA[l]->Fill(data_PCA[l]);

    delete[] input_data;
    delete[] data_PCA;
  }
  double* gauss_means = new double[layer.size()];
  double* gauss_rms   = new double[layer.size()];
 for(unsigned int l=0;l<layer.size();l++)
 {
    gauss_means[l] = h_data_PCA[l]->GetMean();
    gauss_rms[l]   = h_data_PCA[l]->GetRMS();
  }

 if(m_storeDetails)
 {
    TFile* output = TFile::Open( m_outfilename.c_str(), "UPDATE" );
    output->cd( Form( "bin%i/", bin ) );
  for(unsigned int l=0;l<layer.size();l++)
  {
      h_data_PCA[l]->Write( Form( "h_PCA_component%i", l ) );
      histos_data[l]->Write( Form( "h_input_%s", layer[l].c_str() ) );
      cumul_data[l]->Write( Form( "h_cumul_%s", layer[l].c_str() ) );
    }
    // output->Add(T_Gauss);
    T_Gauss->Write();
    output->Write();
    output->Close();
  }

  // cleanup
  delete bintree;
  delete read_bintree;
  delete reader_treeGauss;
  delete reader_treeGauss0;
  if ( !m_storeDetails ) delete T_Gauss;
  delete T_Gauss0;
  delete[] data;
  delete[] data_Gauss;
  delete[] data_Gauss0;
 for (auto it = h_data_PCA.begin(); it != h_data_PCA.end(); ++it)
  delete *it;
  h_data_PCA.clear();

  // get the lower ranges and store them:
  /*
  double* lowerBound=new double[layer.size()];
  for(unsigned int l=0;l<layer.size();l++)
  {
   lowerBound[l]=get_lowerBound(cumul_data[l]);
  }
  */
  // Save EigenValues/EigenVectors/CovarianceMatrix in the output file
  IntArray* myArray = new IntArray( (int)( layer.size() - 1 ) );
  myArray->Set( layer.size() - 1, samplings );

  TMatrixD*    EigenVectors     = (TMatrixD*)principal->GetEigenVectors();
  TMatrixD*    CovarianceMatrix = (TMatrixD*)principal->GetCovarianceMatrix();
  TMatrixDSym* symCov           = new TMatrixDSym();
  symCov->Use( CovarianceMatrix->GetNrows(), CovarianceMatrix->GetMatrixArray() ); // symCov to be stored!

  TVectorD* MeanValues  = (TVectorD*)principal->GetMeanValues();
  TVectorD* SigmaValues = (TVectorD*)principal->GetSigmas();
  TVectorD* Gauss_means = new TVectorD( (int)( layer.size() ), gauss_means );
  TVectorD* Gauss_rms   = new TVectorD( (int)( layer.size() ), gauss_rms );
  // TVectorD* LowerBounds =new TVectorD((int)(layer.size()),lowerBound);

  TFile* output = TFile::Open( m_outfilename.c_str(), "UPDATE" );
  output->cd( Form( "bin%i/pca", bin ) );
  symCov->Write( "symCov" );
  EigenVectors->Write( "EigenVectors" );
  MeanValues->Write( "MeanValues" );
  SigmaValues->Write( "SigmaValues" );
  Gauss_means->Write( "Gauss_means" );
  Gauss_rms->Write( "Gauss_rms" );
  myArray->Write( "RelevantLayers" );
  // LowerBounds ->Write("LowerBounds");
  output->Write();
  output->Close();

  // call the TFCS1DFunctionFactory to decide whether or not to use regression:
 for(unsigned int l=0;l<layer.size();l++)
 {
    cout << endl;
    cout << "====> Now create the fct object for " << layer[l] << " <====" << endl;
    cout << endl;
    stringstream ss;
    ss << bin;
    string          binstring = ss.str();
  TFCS1DFunction* fct=TFCS1DFunctionFactory::Create(cumul_data[l],m_skip_regression,m_neurons_start,m_neurons_end,m_maxdev_regression,m_maxdev_smartrebin,m_ntoys);

    // Store it:
    TFile* output = TFile::Open( m_outfilename.c_str(), "UPDATE" );
    output->cd( Form( "bin%i/%s/", bin, layer[l].c_str() ) );
    fct->Write();
    output->Write();
    output->Close();
  }

} // do_pca


double secondPCA::get_lowerBound(TH1D* h_cumulative)
{
 
 return h_cumulative->GetBinContent(1);
 
}


vector<TH1D*> secondPCA::get_histos_data(vector<string> layer, TreeReader* read_bintree)
{

  vector<TH1D*> data;

  // get the maxima per layer:
  vector<double> MaxInputs;
  for ( unsigned int l = 0; l < layer.size(); l++ ) MaxInputs.push_back( 0.0 );

  vector<double> MinInputs;
  for ( unsigned int l = 0; l < layer.size(); l++ ) MinInputs.push_back( 1000000.0 );

 for(int event=0;event<read_bintree->GetEntries();event++)
 {
    read_bintree->GetEntry( event );
  for(unsigned int l=0;l<layer.size();l++)
  {
      double val = read_bintree->GetVariable( Form( "energy_%s", layer[l].c_str() ) );
   if(val>MaxInputs[l])
    MaxInputs[l]=val;
   if(val<MinInputs[l])
    MinInputs[l]=val;
    }
  }

 for(unsigned int l=0; l<layer.size(); l++)
 {
    TH1D* h_data;
  h_data = new TH1D(Form("h_data_%s",layer[l].c_str()),Form("h_data_%s",layer[l].c_str()),m_numberfinebins,MinInputs[l],MaxInputs[l]);
  for(int event=0;event<read_bintree->GetEntries();event++)
  {
      read_bintree->GetEntry( event );
      h_data->Fill( read_bintree->GetVariable( Form( "energy_%s", layer[l].c_str() ) ) );
    }

    h_data->Sumw2();
    h_data->Scale( 1.0 / h_data->Integral() );
    data.push_back( h_data );

  } // for layer

  return data;
}

vector<int> secondPCA::getLayerBins(TFile* file, int &bins)
{

  vector<int> layer;

  TH2I* h_layer = (TH2I*)file->Get( "h_layer" );

  // the layers are stored in the y axis
 for(int i=1;i<=h_layer->GetNbinsY();i++)
 {
 	if(h_layer->GetBinContent(1,i)==1) 
 	 layer.push_back(h_layer->GetYaxis()->GetBinCenter(i));
  }

  bins = h_layer->GetNbinsX();

  return layer;
}


// double secondPCA::get_cumulant_random(double x, TH1D* h)
double secondPCA::get_cumulant_random(CLHEP::HepRandomEngine *randEngine, double x, TH1D* h)
{

  int bin = h->FindBin( x );

  double content        = h->GetBinContent( bin );
  double before_content = h->GetBinContent( bin - 1 );

  // TRandom3 ran(0);
  // double cumulant=ran.Uniform(before_content,content);
  // cout<<"before_content "<<before_content<<" content "<<content<<" cumulant "<<cumulant<<endl;

  double cumulant = CLHEP::RandFlat::shoot( randEngine, before_content, content );

  return cumulant;
}
