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

#include "TLatex.h"
#include "TH1F.h"
#include "TH2F.h"
#include "TProfile.h"
#include "TStyle.h"
#include "TLegend.h"
#include "TSpline.h"
#include "TF1.h"

#include "TRandom3.h"
#include "TRandom.h"

#include "TGaxis.h"



/////////////////////////////////////////////////////////////////////////////
void X2P(int gNVariables, TMatrixD *EigenVectors,TVectorD *MeanValues, TVectorD *SigmaValues, double *x, double *p) {
    /////////////////////////////////////////////////////////////////////////////
    
    double* gEigenVectors = EigenVectors->GetMatrixArray();
    double* gMeanValues = MeanValues->GetMatrixArray();
    double* gSigmaValues = SigmaValues->GetMatrixArray();
    
    for (int i = 0; i < gNVariables; i++) {
        p[i] = 0;
        for (int j = 0; j < gNVariables; j++)
            p[i] += (x[j] - gMeanValues[j])
            * gEigenVectors[j *  gNVariables + i] / gSigmaValues[j];
    }
}
/////////////////////////////////////////////////////////////////////////////////////////
void P2X(int gNVariables, TMatrixD *EigenVectors,TVectorD *MeanValues, TVectorD *SigmaValues, double *p, double *x, int nTest) {
    /////////////////////////////////////////////////////////////////////////////////////////
    
    double* gEigenVectors = EigenVectors->GetMatrixArray();
    double* gMeanValues = MeanValues->GetMatrixArray();
    double* gSigmaValues = SigmaValues->GetMatrixArray();
    
    for (int i = 0; i < gNVariables; i++) {
        x[i] = gMeanValues[i];
        for (int j = 0; j < nTest; j++)
            x[i] += p[j] * gSigmaValues[i]
            * gEigenVectors[i *  gNVariables + j];
        
    }
}


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
    
    if (y0 == x1) {
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


//////////////////////////////////////////////////////////////////////
void PlotterRatio(TH1F* hinput, TH1F* houtput, std::string output){
//////////////////////////////////////////////////////////////////////
    
    hinput->Rebin(100);
    houtput->Rebin(100);
    
    houtput->Scale(hinput->Integral()/houtput->Integral());
    
    
    //Plotter Style / Draw
	TCanvas *tcv =new TCanvas("Plotter","Plotter",1,1,750,800);
	tcv->SetFillColor(10);
    
    tcv->cd();
    
	TPad *Pad1 = new TPad("Pad1","This is Pad1",0.01,0.20,0.99,0.99);
	TPad *Pad2 = new TPad("Pad2","This is Pad2",0.01,0.01,0.99,0.25);
    
    Pad1->SetLogy(0);
    
	Pad1->SetFillColor(10);
    Pad1->SetBorderMode(0);
    Pad1->SetBorderSize(2);
    Pad1->SetTickx(1);
    Pad1->SetTicky(1);
    Pad1->SetFrameBorderMode(0);
    Pad1->SetFrameFillColor(0);
    Pad1->SetFrameBorderMode(1);
    Pad1->SetLeftMargin(0.15);
    Pad2->SetFillColor(10);
    Pad2->SetTickx(1);
    Pad2->SetTicky(1);
    Pad2->SetGridy(1);
    Pad2->SetLeftMargin(0.15);
    Pad2->SetBottomMargin(0.25);
    Pad1->Draw();
    Pad2->Draw("same");
    
    TLatex *tex = new TLatex();
    tex->SetNDC();
    tex->SetTextFont(72);
    tex->SetLineWidth(2);
    tex->DrawLatex(0.2,0.85,"ATLAS");
    tex->SetTextFont(42);
    
    tex->SetLineWidth(1);
    tex->DrawLatex(0.36,0.85," Internal Simulation");
    
    
    TLegend *legend = new TLegend (0.6,0.65,0.85,0.8);
    legend->SetFillColor(10);
    legend->SetBorderSize(0);
    legend->AddEntry(hinput, "Input dist." , "f");
    legend->AddEntry(houtput, "Output dist." , "lp");
    
	
	Pad1->cd();
    
	hinput->Draw("HIST");
    hinput->SetFillColor(17);
    hinput->SetLineColor(17);
    double max = hinput->GetMaximum();
    hinput->SetMaximum(max*1.5);
    hinput->SetTitle(output.c_str());
    
    
    houtput->Draw("SAME");
    houtput->SetLineColor(kBlue);
    houtput->SetLineWidth(2);
	tex->Draw();
	legend->Draw();
    
	Pad1->Update();
    
    //------------------------
    
    
    
    // Ratio plot
        Pad2->cd();
    
        TH1* hratio = static_cast<TH1*>(houtput->Clone());
    
        hratio->Sumw2();
    
            hratio->Divide(hinput);
            
            // set error on data points to data only (and not total data+MC error)
            /*for(int i=1;i<=nbin;i++)
            {
                hratio->SetBinError(i,data->GetBinError(i)/data->GetBinContent(i));
            }*/
            
            hratio->GetYaxis()->SetTitle("Ratio");
            
            hratio->GetXaxis()-> SetTitle( "" );
            
            hratio->GetYaxis()->SetTitleOffset(0.5);
            hratio->GetYaxis()->SetTitleFont(42);
            hratio->GetYaxis()->SetTitleSize(0.15);
            hratio->GetYaxis()->SetNdivisions(205, "optimized");
            // hratio->GetYaxis()->SetRangeUser(0.5,1.5);
            hratio->GetYaxis()->SetRangeUser(0.8,1.2);
            // 	hratio->GetYaxis()->SetRangeUser(0.,2.0);
            hratio->GetYaxis()->SetLabelFont(42);
            hratio->GetYaxis()->SetLabelSize(0.15);
            
            hratio->GetXaxis()->SetTitleOffset(1.);
            hratio->GetXaxis()->SetTitleFont(42);
            hratio->GetXaxis()->SetTitleSize(0.15);
            hratio->GetXaxis()->SetLabelFont(42);
            hratio->GetXaxis()->SetLabelSize(0.15);	
            
            hratio->SetMarkerStyle(6);
            //hratio->SetFillColor(1);
           // hratio->SetFillStyle(3004);
            hratio->SetStats(0);
            hratio->SetTitle(0);
            
            hratio->Draw("HIST");
    
    
	Pad2->cd();
    //hinput->Divide(houtput);
	//hinput->Draw("E2");
    
	Pad2->Update();

    
    std::string outputname = output+".png";
    tcv->Print(outputname.c_str());
    
    delete tcv;

	
    
}


///////////////////////////////////////////////////////////////////////
void PlotterToy1(TH1F* hinput, TH1F* houtput, std::string output){
///////////////////////////////////////////////////////////////////////
    
    houtput->Scale(hinput->Integral()/houtput->Integral());
    
    //Plotter Style / Draw
    TCanvas *tcv = new TCanvas("Plotter","Plotter",1,1,600,600);
    tcv->SetFillColor(10);
    tcv->SetLogy(0);
    
    tcv->SetBorderMode(0);
    tcv->SetBorderSize(2);
    tcv->SetTickx(1);
    tcv->SetTicky(1);
    tcv->SetFrameBorderMode(0);
    tcv->SetFrameFillColor(0);
    tcv->SetFrameBorderMode(1);
    tcv->SetLeftMargin(0.15);
    
    if (output.find("Id") != std::string::npos) {
        tcv->cd()->SetLogy(1);
    }
    
    else{tcv->cd();}
    
    double KST = houtput->KolmogorovTest(hinput);
    double Chi2 = houtput->Chi2Test(hinput);
    
    hinput->Draw("HIST");
    hinput->SetFillColor(17);
    hinput->SetLineColor(17);
    double max = hinput->GetMaximum();
    hinput->SetMaximum(max*1.5);
    hinput->SetTitle(output.c_str());
    hinput->SetLineWidth(2);
    
    TGaxis::SetMaxDigits(3);
    
    houtput->Draw("SAME");
    houtput->SetLineColor(kBlue);
    houtput->SetLineWidth(2);
    
    TLegend *legend = new TLegend (0.6,0.65,0.85,0.8);
    legend->SetFillColor(10);
    legend->SetBorderSize(0);
    legend->AddEntry(hinput, "Input dist." , "f");
    legend->AddEntry(houtput, "Output dist." , "lp");
    
    legend->Draw();
    
    TLatex *tex = new TLatex();
    tex->SetNDC();
    tex->SetTextFont(72);
    tex->SetLineWidth(2);
    tex->DrawLatex(0.2,0.82,"ATLAS");
    tex->SetTextFont(42);
    tex->SetLineWidth(1);
    tex->DrawLatex(0.36,0.82," Internal Simulation");
    
    tex->DrawLatex(0.60,0.60,Form("#scale[0.70]{KS test : %f }", KST));
    tex->DrawLatex(0.60,0.55,Form("#scale[0.70]{Chi2 test : %f }", Chi2));
    
    gStyle->SetOptStat(0);
    tcv->Update();
    
    std::string outputname = output+".pdf";
    tcv->Print(outputname.c_str());
    
    delete tcv;
    
}

///////////////////////////////////////////////////////////////////////
void PlotterToy2(TH1F* hinput, TH1F* houtput, std::string output){
    ///////////////////////////////////////////////////////////////////////
    
    /*hinput->Rebin(2);
    houtput->Rebin(2);
    */
    
    //Plotter Style / Draw
    TCanvas *tcv = new TCanvas("Plotter","Plotter",1,1,600,600);
    tcv->SetFillColor(10);
    tcv->SetLogy(0);
    
    tcv->SetBorderMode(0);
    tcv->SetBorderSize(2);
    tcv->SetTickx(1);
    tcv->SetTicky(1);
    tcv->SetFrameBorderMode(0);
    tcv->SetFrameFillColor(0);
    tcv->SetFrameBorderMode(1);
    tcv->SetLeftMargin(0.15);
    
    if (output.find("Id") != std::string::npos) {
        tcv->cd()->SetLogy(1);
    }
    
    else{tcv->cd();}
    
    double KST = houtput->KolmogorovTest(hinput);
    double Chi2 = houtput->Chi2Test(hinput);
    
    hinput->Draw("HIST");
    hinput->SetFillColor(17);
    hinput->SetLineColor(17);
    double max = houtput->GetMaximum();
    hinput->SetMaximum(max*1.5);
    hinput->SetTitle(output.c_str());
    
    
    
    houtput->Draw("SAME");
    houtput->SetLineColor(kBlue);
    houtput->SetLineWidth(2);
    
    TLegend *legend = new TLegend (0.6,0.65,0.85,0.8);
    legend->SetFillColor(10);
    legend->SetBorderSize(0);
    
    legend->AddEntry(hinput, "Correlation ignored" , "f");
    legend->AddEntry(houtput, "Including correlations" , "lp");
    
    legend->Draw();
    
    TLatex *tex = new TLatex();
    tex->SetNDC();
    tex->SetTextFont(72);
    tex->SetLineWidth(2);
    tex->DrawLatex(0.2,0.82,"ATLAS");
    tex->SetTextFont(42);
    tex->SetLineWidth(1);
    tex->DrawLatex(0.36,0.82," Internal Simulation");
    
   /* tex->DrawLatex(0.60,0.60,Form("#scale[0.70]{KS test : %f }", KST));
    tex->DrawLatex(0.60,0.55,Form("#scale[0.70]{Chi2 test : %f }", Chi2));
    */
    gStyle->SetOptStat(0);
    tcv->Update();
    
    std::string outputname = output+".png";
    tcv->Print(outputname.c_str());
    
    delete tcv;
    
}

////////////////////////////////////////////////////
void PCA_ToySimulation(TString InputFile="PCAOutput.root", TString InputFile2="PCA_transformation_inputs.root"){
////////////////////////////////////////////////////
    
    //Open Input Files
    TFile *input(0);
    if (!gSystem->AccessPathName( InputFile ))
        input = TFile::Open( InputFile ); // check if file in local directory exists
    if (!input) {
        std::cout << "ERROR: could not open data file" << std::endl;
        exit(1);
    }
    std::cout << " Using input file: " << input->GetName() << std::endl;
    
    TFile *input2(0);
    if (!gSystem->AccessPathName( InputFile2 ))
        input2 = TFile::Open( InputFile2 ); // check if file in local directory exists
    if (!input) {
        std::cout << "ERROR: could not open data file 2" << std::endl;
        exit(1);
    }
    std::cout << " Using input file 2 : " << input2->GetName() << std::endl;
    
    //Input Tree Reader
    TTree *T_Id = (TTree*)input->Get("Id");
    TTree *T_Gauss = (TTree*)input->Get("Gauss");
    TTree *T_PCA = (TTree*)input->Get("PCA");
    
    //--------------------------------------------------------------------------
    //                  Verif. Inverse Cumulant
    //--------------------------------------------------------------------------

    
    //int Layers[] = {0,1,2,3,4,5,6,7,8,12,13,14,15,16,17,18,19};
    int Layers[] = {0,1,2,3,12,13,14};

    std::vector<int> Relevant_Layers (Layers, Layers + sizeof(Layers) / sizeof(int) );;
    
    TreeReader* nr = new TreeReader;
	nr->SetTree(T_Id);
	int NEntries = nr->GetEntries();
    
    TreeReader* nr_Gauss = new TreeReader;
	nr_Gauss->SetTree(T_Gauss);
	int NEntries_Gauss = nr_Gauss->GetEntries();
    
    TreeReader* nr_PCA = new TreeReader;
	nr_PCA->SetTree(T_PCA);
	int NEntries_PCA = nr_PCA->GetEntries();
    
    //Get PCA Transformation
    TMatrixD *EigenVectors = (TMatrixD*)input2->Get("output1_EigenVectors"); //EigenVectors->Print();
    TVectorD *EigenValues = (TVectorD*)input2->Get("output1_EigenValues");
    TVectorD *MeanValues = (TVectorD*)input2->Get("output1_MeanValues");
    TVectorD *SigmaValues = (TVectorD*)input2->Get("output1_SigmaValues");
    
    TH1F** hdata = new TH1F*[Relevant_Layers.size()+1];
    TH1F** hdata_Gauss = new TH1F*[Relevant_Layers.size()+1];
    TH1F** hdata_PCA = new TH1F*[Relevant_Layers.size()+1];

    TH1F** hdata_R = new TH1F*[Relevant_Layers.size()+1];
    TH1F** hdata_R_Gauss = new TH1F*[Relevant_Layers.size()+1];
    TH1F** hdata_R_PCA = new TH1F*[Relevant_Layers.size()+1];
    
    TH1F** hCumulative_Id = new TH1F*[Relevant_Layers.size()];
    TH1F** hCumulative_Id_R = new TH1F*[Relevant_Layers.size()];
    TH1F** hCumulative_PCA = new TH1F*[Relevant_Layers.size()];
    
    TH1F* hTotal_energy_fraction = new TH1F("Total_energy_fraction", "Total_energy_fraction", 100, 0, 2);
    TH1F* hTotal_energy_fraction_R = new TH1F("Total_energy_fraction_R", "Total_energy_fraction_R", 100,0, 2);
    TH1F* hTotal_energy_fraction_R_Dispersion_Test = new TH1F("Total_energy_fraction_R_Dispersion_Test", "Total_energy_fraction_R_Dispersion_Test", 100,0,2);
   
    for (unsigned int k= 0; k<Relevant_Layers.size()+1 ; k++) {
        
        if(k==Relevant_Layers.size()){
            //Energy Fraction Dist.
          /*  hdata[k] = new TH1F("hdata_Total_Cell_Energy","hdata_Total_Cell_Energy",50,T_Id->GetMinimum("hdata_Total_Cell_Energy"),T_Id->GetMaximum("hdata_Total_Cell_Energy"));
            hdata_R[k] = new TH1F("hdata_R_Total_Cell_Energy", "hdata_R_Total_Cell_Energy",50,T_Id->GetMinimum("hdata_Total_Cell_Energy"),T_Id->GetMaximum("hdata_Total_Cell_Energy"));
            */
            
            hdata[k] = new TH1F("hdata_Total_Cell_Energy","hdata_Total_Cell_Energy",50,0,90000);
            hdata_R[k] = new TH1F("hdata_R_Total_Cell_Energy", "hdata_R_Total_Cell_Energy",50,0,90000);

            
            for ( int ientry=0;ientry<NEntries;ientry++ ){
                nr->GetEntry(ientry);
                hdata[k]->Fill(nr->GetVariable("data_Total_Cell_Energy"));
            }
            
            //Gauss Dist.
            hdata_Gauss[k] = new TH1F("hdata_Gauss_Total_Cell_Energy","hdata_Gauss_Total_Cell_Energy",50,T_Gauss->GetMinimum("hdata_Gauss_Total_Cell_Energy"),T_Gauss->GetMaximum("hdata_Gauss_Total_Cell_Energy"));
            hdata_R_Gauss[k] = new TH1F("hdata_R_Gauss_Total_Cell_Energy", "hdata_R_Gauss_Total_Cell_Energy",50,T_Gauss->GetMinimum("hdata_Gauss_Total_Cell_Energy"),T_Gauss->GetMaximum("hdata_Gauss_Total_Cell_Energy"));
            
            for ( int ientry=0;ientry<NEntries_Gauss;ientry++ ){
                nr_Gauss->GetEntry(ientry);
                hdata_Gauss[k]->Fill(nr_Gauss->GetVariable("data_Gauss_Total_Cell_Energy"));
            }
            
            //Cumulatives
            hCumulative_Id[k] = (TH1F*)input->Get("hCumulative_Total_Cell_Energy");
            hCumulative_Id_R[k] = new TH1F("hCumulative_R_Total_Cell_Energy", "hCumulative_R_Total_Cell_Energy",50,T_Id->GetMinimum("hdata_Total_Cell_Energy"),T_Id->GetMaximum("hdata_Total_Cell_Energy"));
            hCumulative_PCA[k] = (TH1F*)input->Get(Form("hCumulative_PCA_%i",k+1));

        }
        
        else{     //Energy Fraction Dist.
            hdata[k] = new TH1F(Form("hdata%i",k+1), Form("hdata%i",k+1),50,T_Id->GetMinimum(Form("data_%i",Relevant_Layers[k])),T_Id->GetMaximum(Form("data_%i",Relevant_Layers[k])));
            hdata_R[k] = new TH1F(Form("hdata_R%i",k+1), Form("hdata_R%i",k+1),50,T_Id->GetMinimum(Form("data_%i",Relevant_Layers[k])),T_Id->GetMaximum(Form("data_%i",Relevant_Layers[k])));
            
            for ( int ientry=0;ientry<NEntries;ientry++ ){
                nr->GetEntry(ientry);
                hdata[k]->Fill(nr->GetVariable(Form("data_%i",Relevant_Layers[k])));
            }
            
            //Gauss Dist.
            hdata_Gauss[k] = new TH1F(Form("hdata_Gauss%i",k+1), Form("hdata_Gauss%i",k+1),50,T_Gauss->GetMinimum(Form("data_Gauss_%i",Relevant_Layers[k])),T_Gauss->GetMaximum(Form("data_Gauss_%i",Relevant_Layers[k])));
            hdata_R_Gauss[k] = new TH1F(Form("hdata_R_Gauss%i",k+1), Form("hdata_R_Gauss%i",k+1),50,T_Gauss->GetMinimum(Form("data_Gauss_%i",Relevant_Layers[k])),T_Gauss->GetMaximum(Form("data_Gauss_%i",Relevant_Layers[k])));
            
            for ( int ientry=0;ientry<NEntries_Gauss;ientry++ ){
                nr_Gauss->GetEntry(ientry);
                hdata_Gauss[k]->Fill(nr_Gauss->GetVariable(Form("data_Gauss_%i",Relevant_Layers[k])));
            }
            
            //Cumulatives
            hCumulative_Id[k] = (TH1F*)input->Get(Form("hCumulative_Id_%i",Relevant_Layers[k]));
            hCumulative_Id_R[k] = new TH1F(Form("hCumulative_Id_R%i",k+1), Form("hCumulative_Id_R%i",k+1),50,T_Id->GetMinimum(Form("data_%i",Relevant_Layers[k])),T_Id->GetMaximum(Form("data_%i",Relevant_Layers[k])));
            hCumulative_PCA[k] = (TH1F*)input->Get(Form("hCumulative_PCA_%i",k+1));
            
        }
        
        
        
        
        
        //PCA Dist.
        hdata_PCA[k] = new TH1F(Form("hdata_PCA%i",k+1), Form("hdata_PCA%i",k+1),50,T_PCA->GetMinimum(Form("PCA_%i",k+1)),T_PCA->GetMaximum(Form("PCA_%i",k+1)) );
        hdata_R_PCA[k] = new TH1F(Form("hdata_R_PCA%i",k+1), Form("hdata_R_PCA%i",k+1),50,T_PCA->GetMinimum(Form("PCA_%i",k+1)),T_PCA->GetMaximum(Form("PCA_%i",k+1)) );
        
        for ( int ientry=0;ientry<NEntries_PCA;ientry++ ){
            nr_PCA->GetEntry(ientry);
            hdata_PCA[k]->Fill(nr_PCA->GetVariable(Form("PCA_%i",k+1)));
        }
        

        

    }
    
    
    //-------------------------------------------------------------------------

    
    for ( int ientry=0;ientry<NEntries;ientry++ ){
        
        nr->GetEntry(ientry);
        double data[Relevant_Layers.size()];
        double Total_energy_fraction = 0 ;

        for (unsigned int i=0; i<Relevant_Layers.size(); i++) {
            data[i] = nr->GetVariable(Form("data_%i",Relevant_Layers[i]));
            
            Total_energy_fraction = Total_energy_fraction + data[i];
            
        }
        
        hTotal_energy_fraction->Fill(Total_energy_fraction);

        //hTotal_energy_fraction->Fill(nr->GetVariable("Total_energy_fraction"));
        
    }
    
    
    //-----------------------------------------------------------------------------------------------
    //    (2) Simulation chain
    //-----------------------------------------------------------------------------------------------
    
    int NPseudoExp = 10000;
    
    for (int i=0; i<NPseudoExp; i++) {
    
        TRandom3* Random = new TRandom3();
        Random->SetSeed(i);
    
        double Simulated_Uniform[Relevant_Layers.size()+1];
        double Simulated_PCA[Relevant_Layers.size()+1];
        double Simulated_Gauss[Relevant_Layers.size()+1];
        double Simulated_Uniform2[Relevant_Layers.size()+1];
        double Simulated_Event[Relevant_Layers.size()+1];
        double Dispersion_Test[Relevant_Layers.size()];

        
        for (unsigned int k=0; k<Relevant_Layers.size()+1; k++) {
            Simulated_Uniform[k] = Random->Uniform(1); //std::cout<<Random->Uniform(1) <<"\t" ;
            Simulated_PCA[k] = InverseCumulant(Simulated_Uniform[k], hCumulative_PCA[k]) ;
            hdata_R_PCA[k]->Fill(Simulated_PCA[k]);
        }
        
        
        if (i%1000 == 0 ) {
            std::cout<<i <<" sumulated events" <<std::endl;
        }
    
        P2X(Relevant_Layers.size()+1, EigenVectors, MeanValues, SigmaValues, Simulated_PCA, Simulated_Gauss, Relevant_Layers.size()+1);

        double Tampon_total_energy_fraction = 0;
        for (unsigned int k=0; k< Relevant_Layers.size()+1; k++) {
            
            hdata_R_Gauss[k]->Fill(Simulated_Gauss[k]);
            Simulated_Uniform2[k] = (TMath::Erf(Simulated_Gauss[k]/1.414213562)+1)/2.f;
            Simulated_Event[k] = InverseCumulant(Simulated_Uniform2[k], hCumulative_Id[k]) ;
            hdata_R[k]->Fill(Simulated_Event[k]);
          
        }
   
        double Total_energy_fraction_R = 0 ;
        double Total_energy_fraction_R_Dispersion = 0 ;
        
        
        for (unsigned int k=0; k<Relevant_Layers.size(); k++) {
            Dispersion_Test[k] = InverseCumulant(Simulated_Uniform[k], hCumulative_Id[k]) ;
            Total_energy_fraction_R = Total_energy_fraction_R + Simulated_Event[k];
            Total_energy_fraction_R_Dispersion = Total_energy_fraction_R_Dispersion + Dispersion_Test[k];
        }
        
        hTotal_energy_fraction_R->Fill(Total_energy_fraction_R);
        hTotal_energy_fraction_R_Dispersion_Test->Fill(Total_energy_fraction_R_Dispersion);

    }
  
    //-----------------------------------------------------------------------------------------------
    //    (3) Validation
    //-----------------------------------------------------------------------------------------------

    // Control Plots
    for (unsigned int k=0; k<Relevant_Layers.size()+1; k++) {
        
        if(k==Relevant_Layers.size()){
            
            std::string outputname = "ToySimulation2_Var_Total_Cell_Energy";
            PlotterToy1(hdata[k], hdata_R[k], outputname);
            
            std::string outputname2 = "ToySimulation2_Gauss_Total_Cell_Energy";
            PlotterToy1(hdata_Gauss[k], hdata_R_Gauss[k], outputname2);}
        
        else{
        std::string outputname = Form("ToySimulation2_Var%i",Relevant_Layers[k]);
        PlotterToy1(hdata[k], hdata_R[k], outputname);
        
        std::string outputname2 = Form("ToySimulation2_Gauss%i",Relevant_Layers[k]);
        PlotterToy1(hdata_Gauss[k], hdata_R_Gauss[k], outputname2);
        
        }
        
        std::string outputname3 = Form("ToySimulation2_PCA%i",k);
        PlotterToy1(hdata_PCA[k], hdata_R_PCA[k], outputname3);
        
        //Cumulative comparison
        hdata_R[k]->Sumw2();
        hdata_R[k]->Scale(1/hdata_R[k]->Integral());
        //std::cout<<hdata_R[k]->GetNbinsX() <<std::endl;
        for (int i=1; i<=hdata_R[k]->GetNbinsX(); i++) {
            hCumulative_Id_R[k]->SetBinContent(i,hdata_R[k]->Integral(1,i));
        }
        
        std::string outputname6 = Form("ToySimulation2_Ratio_Cumulative_Var%i",Relevant_Layers[k]);
       // PlotterToy1( hCumulative_Id[k],  hCumulative_Id_R[k], outputname6);
       // PlotterRatio( hCumulative_Id[k],  hCumulative_Id_R[k], outputname6);
    }
                                         
        std::string outputname4 = "ToySimulation2_Total_energy_fraction";
        PlotterToy1(hTotal_energy_fraction, hTotal_energy_fraction_R, outputname4);
    
        std::string outputname5 = "ToySimulation2_Total_energy_fraction_Dispersion_Test";
        PlotterToy1(hTotal_energy_fraction, hTotal_energy_fraction_R_Dispersion_Test, outputname5);
    
        std::string outputname7 = "ToySimulation2_Total_energy_fraction_Dispersion_Test2";
        PlotterToy2(hTotal_energy_fraction_R_Dispersion_Test, hTotal_energy_fraction_R , outputname5);

    
    
    std::cout<<"hTotal_energy_fraction : " <<hTotal_energy_fraction->GetRMS() <<"+/-" <<hTotal_energy_fraction->GetRMSError() <<std::endl;
    std::cout<<"hTotal_energy_fraction_R : " <<hTotal_energy_fraction_R->GetRMS() <<"+/-" <<hTotal_energy_fraction_R->GetRMSError() <<std::endl;
    std::cout<<"hTotal_energy_fraction_R_Dispersion_Test : " <<hTotal_energy_fraction_R_Dispersion_Test->GetRMS() <<"+/-" <<hTotal_energy_fraction_R_Dispersion_Test->GetRMSError() <<std::endl;
    

    std::cout<<"hTotal_Cell_energy : " <<hdata[Relevant_Layers.size()]->GetMean() <<"+/-" <<hdata[Relevant_Layers.size()]->GetMeanError() <<"    -    " <<hdata[Relevant_Layers.size()]->GetRMS() <<"+/-" <<hdata[Relevant_Layers.size()]->GetRMSError() <<std::endl;
 //   std::cout<<"hTotal_Cell_energy_R : " <<hTotal_energy_fraction_R->GetRMS() <<"+/-" <<hTotal_energy_fraction_R->GetRMSError() <<std::endl;
    
    
    std::cout<<"hTotal_Cell_energy_R : " <<hdata_R[Relevant_Layers.size()]->GetMean()<<"+/-" <<hdata[Relevant_Layers.size()]->GetMeanError() <<"    -    " <<hdata_R[Relevant_Layers.size()]->GetRMS() <<"+/-" <<hdata_R[Relevant_Layers.size()]->GetRMSError() <<std::endl;
   
    
}












