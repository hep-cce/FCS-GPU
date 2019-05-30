/*
  Copyright (C) 2002-2017 CERN for the benefit of the ATLAS collaboration
*/

#include "TTree.h"
#include "TString.h"
#include <string>
#include <sstream>
#include <iostream>
#include "TSystem.h"
#include "TString.h"
#include "TFile.h"
#include "TH1F.h"
#include <stdlib.h>

#include "TreeReader.h"
#include "TPrincipal.h"
#include "TMath.h"
#include "TBrowser.h"

#include "TGraph.h"
#include "TCanvas.h"

#include "TMatrixD.h"
#include "TVectorD.h"

//////////////////////////////////////////////////////////
void quantiles(TH1F* h, int nq, double* xq, double* yq) {
//////////////////////////////////////////////////////////
    
    //Function for quantiles
    // xq position where to compute the quantiles in [0,1]
    // yq array to contain the quantiles
    
    for (Int_t i=0;i<nq;i++){
        xq[i] = float(i+1)/nq;
        h->GetQuantiles(nq,yq,xq);
    }
    
}

////////////////////////////////////////////
double Cumulant(float x, TH1F* h){
    ////////////////////////////////////////////
    
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

////////////////////////////////////////////
double InverseCumulant(float y, TH1F* h){
    ////////////////////////////////////////////
    
    int bin;
    int nbin = h->GetNbinsX();
    double min = 99999999;
    for (int i=1; i<=h->GetNbinsX()-2; i++) {
        if(fabs(h->GetBinContent(i)-y)<min){
            min = fabs(h->GetBinContent(i)-y);
            bin = i ;
            
        }
    }
    bin = TMath::Max(bin,1);
    bin = TMath::Min(bin,h->GetNbinsX());
    
    //std::cout<<bin <<std::endl;
    
    
    double AvNuEvPerBin;
    double Tampon = 0 ;
    for (int i=1; i<=nbin; i++) {
        Tampon += h->GetBinContent(i);
    }
    
    AvNuEvPerBin = Tampon/nbin;
    
    double x;
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
    
    if (y0 == y1) {
        // cumulant = y1;
        x = x0;
    } else {
        //cumulant = y0 + (y1-y0)*(x-x0)/(x1-x0);
        x = x0 + (y-y0)*(x1-x0)/(y1-y0);
        
    }
    
    /* if (x <= h->GetBinLowEdge(1)){
     cumulant = supmin;
     }
     if (x >= h->GetBinLowEdge(h->GetNbinsX()+1)){
     cumulant = 1-supmin;
     }*/
    
    return x;
    
}


////////////////////////////////////////////////////
void PCA(TString InputFile="output1"){
////////////////////////////////////////////////////

    //Open Input File
    TFile *input(0);
    TString InputFileName = InputFile+".root";
    if (!gSystem->AccessPathName( InputFileName ))
        input = TFile::Open( InputFileName ); // check if file in local directory exists
    if (!input) {
        std::cout << "ERROR: could not open data file" << std::endl;
        exit(1);
    }
    std::cout << " Using input file: " << input->GetName() << std::endl;
    
    //Input Tree Reader
    TTree *InputTree = (TTree*)input->Get("FCS_ParametrizationInput");
    TreeReader* nr = new TreeReader;
	nr->SetTree(InputTree);
	int NEntries = nr->GetEntries();
    
    
    //-----------------------------------------------------------------------------------------------
    //    (1) Set optimized input histogram size and select relevant layers
    //-----------------------------------------------------------------------------------------------
    
    double MaxInputs[25];
    double Tampon[25];
    for (int i=0; i<25; i++) {
        MaxInputs[i]=0;
        Tampon[i]=0;
    }
    
    for ( int ientry=0;ientry<NEntries;ientry++ ){
        nr->GetEntry(ientry);
        for (int j=0 ; j<25 ; j++) {
            double data = nr->GetVariable(Form("cell_energy[%d]",j))/nr->GetVariable("total_cell_energy");
            Tampon[j] += data;
            
            if(data>MaxInputs[j]){MaxInputs[j]=data;}
        }
        
    }
    
    //Select Relevant Layer
    std::vector<int> Relevant_Layers;
    std::cout<<std::endl <<"- Select Relevant Calo Layers" <<std::endl <<std::endl;
    for (int i=0; i<25 ; i++) {
        if(Tampon[i]/NEntries >= 0.01){
            
            Relevant_Layers.push_back(i);
            
            std::cout<<"Layer "  <<i <<" is relevant !" <<std::endl;
        }
    }

    
    //-----------------------------------------------------------------------------------------------
    //    (2) Fill Input Histograms and Compute Cumulative Distributions
    //-----------------------------------------------------------------------------------------------
    
    TH1F** hdata = new TH1F*[Relevant_Layers.size()+1];
    TH1F** hCumulative = new TH1F*[Relevant_Layers.size()+1];
    
    for (unsigned int k= 0; k < Relevant_Layers.size()+1; k++) {
        
        if (k==Relevant_Layers.size()) {
            hdata[Relevant_Layers.size()] = new TH1F("hdata_Total_Cell_Energy", "hdata_Total_Cell_Energy",5000,InputTree->GetMinimum("total_cell_energy"),InputTree->GetMaximum("total_cell_energy"));
            
            for ( int ientry=0;ientry<NEntries;ientry++ ){
                nr->GetEntry(ientry);
                hdata[k]->Fill(nr->GetVariable("total_cell_energy"));
            }
            //Get Cumulative
            hCumulative[k] = new TH1F("hCumulative_Total_Cell_Energy", "hCumulative_Total_Cell_Energy", 5000,InputTree->GetMinimum("total_cell_energy"),InputTree->GetMaximum("total_cell_energy"));
            
            }
        
        else{
            hdata[k] = new TH1F(Form("hdata%i",Relevant_Layers[k]), Form("hdata%i",Relevant_Layers[k]),5000,0,MaxInputs[Relevant_Layers[k]]);
        
        
            for ( int ientry=0;ientry<NEntries;ientry++ ){
                nr->GetEntry(ientry);
                hdata[k]->Fill( nr->GetVariable(Form("cell_energy[%d]",Relevant_Layers[k]))/nr->GetVariable("total_cell_energy") );
            }

        
            //Get Cumulative
            hCumulative[k] = new TH1F(Form("hCumulative_Id_%i",Relevant_Layers[k]), Form("hCumulative_Id_%i",Relevant_Layers[k]), 5000, 0,MaxInputs[Relevant_Layers[k]] );
    
        }
        
        
    
        
        hdata[k]->Sumw2();
        hdata[k]->Scale(1/hdata[k]->Integral());
        
        for (int i=1; i<=hdata[k]->GetNbinsX(); i++) {
            hCumulative[k]->SetBinContent(i,hdata[k]->Integral(1,i));
        }
        

        
        
        
    }
    
    
   
    //-----------------------------------------------------------------------------------------------
    //    (3) Performe PCA on gaussianized energy fraction distribtuions only for relevant leayers.
    //-----------------------------------------------------------------------------------------------
    
    
    //Get Data
    Double_t* data = new Double_t[Relevant_Layers.size()+1];
    Double_t* data_Uniform = new Double_t[Relevant_Layers.size()+1];
    Double_t* data_Gauss = new Double_t[Relevant_Layers.size()+1];
    
    //Output file
    TFile f("PCAOutput.root","recreate");
    
    TTree *T = new TTree("Id","Energy Factions");
    TTree *T_Uniform = new TTree("Uniform","Uniform");
    TTree *T_Gauss = new TTree("Gauss","gauss");
    
    for (unsigned int k= 0; k<Relevant_Layers.size()+1; k++) {
        
        if(k==Relevant_Layers.size()){
            T->Branch("data_Total_Cell_Energy",&data[k],"data_Total_Cell_Energy/D");
            T_Uniform->Branch("data_Uniform_Total_Cell_Energy",&data_Uniform[k],"data_Uniform_Total_Cell_Energy/D");
            T_Gauss->Branch("data_Gauss_Total_Cell_Energy",&data_Gauss[k],"data_Gauss_Total_Cell_Energy/D");
            }
        else{
            T->Branch(Form("data_%i",Relevant_Layers[k]),&data[k],Form("data_%i/D",Relevant_Layers[k]));
            T_Uniform->Branch(Form("data_Uniform_%i",Relevant_Layers[k]),&data_Uniform[k],Form("data_Uniform_%i/D",Relevant_Layers[k]));
            T_Gauss->Branch(Form("data_Gauss_%i",Relevant_Layers[k]),&data_Gauss[k],Form("data_Gauss_%i/D",Relevant_Layers[k]));

        }
        
       
        //Store Cumulative Distributions
        hCumulative[k]->Write();
    }
    
    //PCA via TPrincipal
    TPrincipal* principal = new TPrincipal(Relevant_Layers.size()+1,"ND");
    
    
    //Uniformization/Gaussianization - Loop on Events
    for ( int ientry=0;ientry<NEntries;ientry++ ){
        
        nr->GetEntry(ientry);
        
        
    //    //Store total energy fraction deposited
    //    double Tampon_total_energy = 0;
    //    for (int j=0 ; j<25 ; j++) {
    ////        Tampon_total_energy = Tampon_total_energy + nr->GetVariable(Form("cell_energy[%d]",j));
    //    }
    //    Total_cell_energy = Tampon_total_energy /nr->GetVariable("total_cell_energy") ;

        for (unsigned int k= 0; k<Relevant_Layers.size()+1; k++) {
            
            if(k==Relevant_Layers.size()){data[k] = nr->GetVariable("total_cell_energy");}
            else{data[k] = nr->GetVariable(Form("cell_energy[%d]",Relevant_Layers[k]))/nr->GetVariable("total_cell_energy");}
            
            //Uniformization
            double cumulant = Cumulant(data[k],hCumulative[k]);
            cumulant = TMath::Min(cumulant,1.-10e-10);
            cumulant = TMath::Max(cumulant,0.+10e-10);
            data_Uniform[k] = cumulant;
            
            //Gaussianization
            double maxErfInvArgRange = 0.99999999;
            double arg = 2.0*cumulant - 1.0;
            arg = TMath::Min(+maxErfInvArgRange,arg);
            arg = TMath::Max(-maxErfInvArgRange,arg);

            data_Gauss[k] = 1.414213562*TMath::ErfInverse(arg) ;

        }
        
        //Add this datapoint to the PCA
        principal->AddRow(data_Gauss);
        
        
        //FIll Output trees
        T->Fill();
        T_Uniform->Fill();
        T_Gauss->Fill();
 
    }
    
    // Do the actual analysis
    principal->MakePrincipals();
    
    //Print out the result on
    std::cout<<std::endl <<"- Principal Component Analysis Results" <<std::endl <<std::endl;
    principal->Print("MSE");
    
    
    //Get EigenValues/EigenVectors/CovarianceMatrix
    TMatrixD* CovarianceMatrix =(TMatrixD*)principal->GetCovarianceMatrix();
    TVectorD* EigenValues =(TVectorD*)principal->GetEigenValues();
    TMatrixD* EigenVectors =(TMatrixD*)principal->GetEigenVectors();
    TVectorD* MeanValues =(TVectorD*)principal->GetMeanValues();
    TVectorD* SigmaValues =(TVectorD*)principal->GetSigmas();
    
    TString Transformation_id = InputFile;
    
    
    //Get Ratio 2 1stPC
    const double *Array_EigenValue = EigenValues->GetMatrixArray();
    int Ratio2_1stPC = round(Array_EigenValue[0]/Array_EigenValue[1])*2 ;
    
    // Test the PCA
    //principal->Test();
    
    
    // Make some histograms of the orginal, principal, residue, etc data
    //principal->MakeHistograms();
    
    
    // Make two functions to map between feature and pattern space
    //principal->MakeCode();
    
    //TBrowser* b = new TBrowser("principalBrowser", principal);
    
    
    delete data;
    delete data_Uniform;
    delete data_Gauss;
    
    delete hdata;
    delete hCumulative;
    
    //-----------------------------------------------------------------------------------------------
    //    (4) PCA application and binning of first principal component
    //-----------------------------------------------------------------------------------------------
    
    TreeReader* nr_Gauss = new TreeReader;
	nr_Gauss->SetTree(T_Gauss);
	int NEntries_Gauss = nr_Gauss->GetEntries();
    
    Double_t* data_PCA = new Double_t[Relevant_Layers.size()+1];
    Double_t* input_data = new Double_t[Relevant_Layers.size()+1];
    
    TTree *T_PCA = new TTree("PCA","PCA");
    int Bin_1stPC;
    TH1F* hPCA_first_composant = new TH1F("hPCA_first_component","hPCA_first_component",5000,-10,10);
    
    T_PCA->Branch("Bin_1stPC",&Bin_1stPC,"Bin_1stPC/I");
    for (unsigned int k= 0; k<Relevant_Layers.size()+1; k++) {
        T_PCA->Branch(Form("PCA_%i",k+1),&data_PCA[k],Form("PCA_%i/D",k+1));
        }
    
    //PCA application on gaussian distributions and binning 1st principal component

    for ( int ientry=0;ientry<NEntries_Gauss;ientry++ ){
        nr_Gauss->GetEntry(ientry);
        for (unsigned int k= 0; k<Relevant_Layers.size()+1; k++) {
            if(k==Relevant_Layers.size()){input_data[k] = nr_Gauss->GetVariable("data_Gauss_Total_Cell_Energy");}
            else{input_data[k] = nr_Gauss->GetVariable(Form("data_Gauss_%i",Relevant_Layers[k]));}
        }
        
        principal->X2P(input_data,data_PCA);
        hPCA_first_composant->Fill(data_PCA[0]);
    }
    
    //Test 10Bins
    //Ratio2_1stPC = 10;
    
    double xq[Ratio2_1stPC];
    double yq[Ratio2_1stPC];
    
    quantiles( hPCA_first_composant, Ratio2_1stPC, xq , yq );
    std::cout<<std::endl <<"- Binning 1st Principal Component" <<std::endl <<std::endl;
    for (int m = 0; m < Ratio2_1stPC ; m++) {
        std::cout<<"Quantiles : " <<m+1 <<"  |  "  <<xq[m] <<"  |  " <<yq[m] <<endl;
    }
    
    for ( int ientry=0;ientry<NEntries_Gauss;ientry++ ){
        nr_Gauss->GetEntry(ientry);
        
        for (unsigned int k= 0; k<Relevant_Layers.size()+1; k++) {
            if(k==Relevant_Layers.size()){input_data[k] = nr_Gauss->GetVariable("data_Gauss_Total_Cell_Energy");}
            else{input_data[k] = nr_Gauss->GetVariable(Form("data_Gauss_%i",Relevant_Layers[k]));}
        }
        
        //PCA Application
        principal->X2P(input_data,data_PCA);
        
        //Binning 1st PC
        for (int m = 0 ; m < Ratio2_1stPC ; m++) {
            if ( m==0 && data_PCA[0] <= yq[m]) {
                Bin_1stPC = m+1;
            }
            else if( m > 0 && data_PCA[0] > yq[m-1] && data_PCA[0] <= yq[m]){
                 Bin_1stPC = m+1 ;
            }
            
        }
        T_PCA->Fill();

    }


    //Write output trees
    T->Write();
    T_Uniform->Write();
    T_Gauss->Write();
    T_PCA->Write();
    
    
    //-----------------------------------------------------------------------------------------------
    //    (5) Get Cumulative Distributions from principal components.
    //-----------------------------------------------------------------------------------------------

    
    TH1F** hdata_PCA = new TH1F*[Relevant_Layers.size()+1];
    TH1F** hCumulative_PCA = new TH1F*[Relevant_Layers.size()+1];
   
    std::vector<TH1F**> hCumulative_PCA_Bin(Relevant_Layers.size()+1);
    std::vector<TH1F**> hdata_PCA_Bin(Relevant_Layers.size()+1);
    for(unsigned int i=0;i<Relevant_Layers.size()+1;i++){
        hCumulative_PCA_Bin[i] = new TH1F*[Ratio2_1stPC];
        hdata_PCA_Bin[i] = new TH1F*[Ratio2_1stPC];
    }
    
    TreeReader* nr_PCA = new TreeReader;
	nr_PCA->SetTree(T_PCA);
	int NEntries_PCA = nr_PCA->GetEntries();
    
    
    for (unsigned int k= 0; k < Relevant_Layers.size()+1 ; k++) {
        
        //Declare cumulatives for principal component
        hdata_PCA[k] = new TH1F(Form("hdata_PCA%i",k+1), Form("hdata_PCA%i",k+1),5000,T_PCA->GetMinimum(Form("PCA_%i",k+1)),T_PCA->GetMaximum(Form("PCA_%i",k+1)));
       
        //Declare cumulatives for binned principal components
        for (int i = 0; i<Ratio2_1stPC; i++) {
           /* if (i==0) {
                hdata_PCA_Bin[k][i] = new TH1F(Form("hdata_PCA%i_bin1st_%i",k+1,i), Form("hdata_PCA%i_bin1st_%i",k+1,i),5000,T_PCA->GetMinimum(Form("PCA_%i",k+1)),yq[i]);
            }
            else{hdata_PCA_Bin[k][i] = new TH1F(Form("hdata_PCA%i_bin1st_%i",k+1,i), Form("hdata_PCA%i_bin1st_%i",k+1,i),5000,yq[i-1],yq[i]);}*/
            
            hdata_PCA_Bin[k][i] = new TH1F(Form("hdata_PCA%i_bin1st_%i",k+1,i+1), Form("hdata_PCA%i_bin1st_%i",k+1,i+1), 5000, T_PCA->GetMinimum(Form("PCA_%i",k+1)),T_PCA->GetMaximum(Form("PCA_%i",k+1)));
        }
        
        //Fill hitograms with principal component data
        for ( int ientry=0;ientry<NEntries_PCA;ientry++ ){
            nr_PCA->GetEntry(ientry);
            hdata_PCA[k]->Fill(nr_PCA->GetVariable(Form("PCA_%i",k+1)));
            
            for (int i = 0; i<Ratio2_1stPC; i++) {
               // std::cout<< nr_PCA->GetVariable("Bin_1stPC") <<std::endl;
                if (nr_PCA->GetVariable("Bin_1stPC") == i+1) {
                    hdata_PCA_Bin[k][i]->Fill(nr_PCA->GetVariable(Form("PCA_%i",k+1)));
                    //std::cout<<"Integral : " <<hdata_PCA_Bin[k][i]->Integral()  <<std::endl;
                }
              
            }
            
        }
        
        //Normalization
        hdata_PCA[k]->Sumw2();
        hdata_PCA[k]->Scale(1/hdata_PCA[k]->Integral());
        
        for (int i = 0; i<Ratio2_1stPC; i++) {
            hdata_PCA_Bin[k][i]->Sumw2();
            hdata_PCA_Bin[k][i]->Scale(1/hdata_PCA_Bin[k][i]->Integral());
            //std::cout<<hdata_PCA_Bin[k][i]->Integral() <<"\t";
            
            hdata_PCA_Bin[k][i]->Write();
        }
        
        
        //Get Cumulatives
        hCumulative_PCA[k] = new TH1F(Form("hCumulative_PCA_%i",k+1), Form("hCumulative_PCA_%i",k+1), 5000, T_PCA->GetMinimum(Form("PCA_%i",k+1)),T_PCA->GetMaximum(Form("PCA_%i",k+1)));
        
        for (int i=1; i<=hdata_PCA[k]->GetNbinsX(); i++) {
            hCumulative_PCA[k]->SetBinContent(i,hdata_PCA[k]->Integral(1,i));
        }
        hCumulative_PCA[k]->Write();
     
        //Getcumulatives for binned principal components
        for (int i = 0; i<Ratio2_1stPC; i++) {
                hCumulative_PCA_Bin[k][i] = new TH1F(Form("hCumulative_PCA%i_bin1st_%i",k+1,i+1), Form("hCumulative_PCA%i_bin1st_%i",k+1,i+1),5000,T_PCA->GetMinimum(Form("PCA_%i",k+1)),T_PCA->GetMaximum(Form("PCA_%i",k+1)));
            
            for (int j=1; j<=hdata_PCA_Bin[k][i]->GetNbinsX(); j++) {
                hCumulative_PCA_Bin[k][i]->SetBinContent(j,hdata_PCA_Bin[k][i]->Integral(1,j));
            }
            
            if (hCumulative_PCA_Bin[k][i]->Integral() > 0) {
                hCumulative_PCA_Bin[k][i]->Write();
            }
        }
        
      //  std::cout<<std::endl;
      
    }
    
    delete nr;
    delete nr_Gauss;
    delete nr_PCA;
    
    delete T;
    delete T_Uniform;
    delete T_Gauss;
    delete T_PCA;
    
    delete data_PCA;
    delete input_data;
    delete hPCA_first_composant;
    
    delete hdata_PCA;
    delete hCumulative_PCA;
   
    
    f.Close();
                               
    
    //-----------------------------------------------------------------------------------------------
    //    (6) Save EigenValues/EigenVectors/CovarianceMatrix
    //-----------------------------------------------------------------------------------------------
    
    TFile f2("PCA_transformation_inputs.root","recreate");
    
    TString Name_CovarianceMatrix = Transformation_id+"_CovarianceMatrix";
    TString Name_EigenValues = Transformation_id+"_EigenValues";
    TString Name_EigenVectors = Transformation_id+"_EigenVectors";
    TString Name_MeanValues = Transformation_id+"_MeanValues";
    TString Name_SigmaValues = Transformation_id+"_SigmaValues";
   
    CovarianceMatrix->Write(Name_CovarianceMatrix);
    EigenValues->Write(Name_EigenValues);
    EigenVectors->Write(Name_EigenVectors);
    MeanValues->Write(Name_MeanValues);
    SigmaValues->Write(Name_SigmaValues);
    
    
    f2.Close();
   

}







