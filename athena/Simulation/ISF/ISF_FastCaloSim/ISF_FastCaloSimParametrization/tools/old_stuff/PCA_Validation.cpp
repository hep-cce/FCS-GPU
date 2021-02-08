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

#include "TCanvas.h"
#include "TLatex.h"
#include "TGraph.h"
#include "TH1F.h"
#include "TH2F.h"
#include "TProfile.h"
#include "TStyle.h"
#include "TLegend.h"
#include "TSpline.h"
#include "TF1.h"

#include "TRandom3.h"

#include "TMatrixD.h"
#include "TVectorD.h"

#include "TRandom.h"


//#include "PCA_Transformation.h"

#define XXX std::cout << "I am here: " << __FILE__ << ":" << __LINE__ << std::endl;

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

///////////////////////////////////////////////////////////////////////
void ScatterPlots(TTree *InputTree, std::string VarPlotted, std::string output){
///////////////////////////////////////////////////////////////////////
    
    
    //Plotter Style / Draw
    TCanvas *tcv = new TCanvas("Plotter","Plotter",1,1,600,600);
    //tcv->Divide(3,1);
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
    
    tcv->cd();
    InputTree->Draw(VarPlotted.c_str(),"","");
    
    TGraph *Graph = (TGraph*)gPad->GetPrimitive("Graph");
    Graph->SetMarkerStyle(6);
    Graph->SetMarkerColor(17);
    //Graph->Fit("pol5");
    
    InputTree->Draw(VarPlotted.c_str(),"","profsame");
    
    TH2F *htemp = (TH2F*)gPad->GetPrimitive("htemp");
    htemp->GetYaxis()->SetTitleOffset(1.4);
    htemp->SetTitle(0);
    htemp->SetMarkerColor(3);
    
    
    TLatex *tex = new TLatex();
    tex->SetNDC();
    tex->SetTextFont(72);
    tex->SetLineWidth(2);
    tex->DrawLatex(0.2,0.82,"ATLAS");
    tex->SetTextFont(42);
    tex->SetLineWidth(1);
    tex->DrawLatex(0.36,0.82," Internal Simulation");
    
    tcv->Update();
    
    std::string outputname = output+".png";
    tcv->Print(outputname.c_str());
    
    delete tcv;
    
}

///////////////////////////////////////////////////////////////////////
void PlotterToy1(TH1F* hinput, TH1F* houtput, std::string output){
///////////////////////////////////////////////////////////////////////


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
    
    
hinput->Draw("HIST");
hinput->SetFillColor(17);
hinput->SetLineColor(17);
//double max = hinput->GetMaximum();
//hinput->SetMaximum(max*1.1);
hinput->SetTitle(output.c_str());

    
houtput->Draw("SAME");
houtput->SetLineColor(kBlue);
    
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
    
gStyle->SetOptStat(0);
tcv->Update();
  
std::string outputname = output+".png";
tcv->Print(outputname.c_str());
    
delete tcv;

}


///////////////////////////////////////////////////////////////////////
void TransformationChainPlots(TTree *InputTree_Id, std::string VarPlotted_Id, TTree *InputTree_Uniform, std::string VarPlotted_Uniform, TTree *InputTree_Gauss, std::string VarPlotted_Gauss, TH1F* Cumulative , std::string output){
///////////////////////////////////////////////////////////////////////
    
    
    //Plotter Style / Draw
    TCanvas *tcv = new TCanvas("Plotter","Plotter",1,1,1200,1200);
    tcv->Divide(2,2);
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
    
    TLatex *tex = new TLatex(0.2,0.82,"ATLAS");
    tex->SetNDC();tex->SetTextFont(72); tex->SetLineWidth(2);
    TLatex *tex2 = new TLatex(0.36,0.82," Internal Simulation");
    tex2->SetNDC(); tex2->SetTextFont(42);tex2->SetLineWidth(1);

    tcv->cd(1)->SetLogy(1);
    InputTree_Id->Draw(VarPlotted_Id.c_str(),"","");
    TH1F *htemp = (TH1F*)gPad->GetPrimitive("htemp");
    htemp->SetTitle("Energy Fraction");
    htemp->SetLineColor(kBlue);
    tex->Draw();
    tex2->Draw();
    gStyle->SetOptStat(0);
    
    tcv->cd(2);
    Cumulative->Draw("HIST");
    Cumulative->SetTitle("Cumulative Distribution");
    Cumulative->SetLineColor(kBlue);
    tex->Draw();
    tex2->Draw();
    gStyle->SetOptStat(0);
    
    tcv->cd(3);
    InputTree_Uniform->Draw(VarPlotted_Uniform.c_str(),"","");
    TH1F *htemp2 = (TH1F*)gPad->GetPrimitive("htemp");
    htemp2->SetTitle("Uniformisation");
    htemp2->SetLineColor(kBlue);
    tex->Draw();
    tex2->Draw();
    gStyle->SetOptStat(0);
    
    tcv->cd(4);
    InputTree_Gauss->Draw(VarPlotted_Gauss.c_str(),"","");
    TH1F *htemp3 = (TH1F*)gPad->GetPrimitive("htemp");
    htemp3->SetTitle("Gaussianisation");
    htemp3->SetLineColor(kBlue);
    tex->Draw();
    tex2->Draw();
    gStyle->SetOptStat(0);

    tcv->Update();
    
    std::string outputname = output+".png";
    tcv->Print(outputname.c_str());
    
    delete tcv;
    
}



////////////////////////////////////////////////////
void ToySimulation1(TFile* input, TFile* input2){
////////////////////////////////////////////////////

    //Input Tree Reader
    TTree *T_Id = (TTree*)input->Get("Id");
    TTree *T_Uniform = (TTree*)input->Get("Uniform");
    TTree *T_Gauss = (TTree*)input->Get("Gauss");
    TTree *T_PCA = (TTree*)input->Get("PCA");
    
    // Input/Output Histogram :
    TH1F** hinputdata = new TH1F*[T_PCA->GetNbranches()];
    TH1F** hinputdataUniform = new TH1F*[T_PCA->GetNbranches()];
    TH1F** hinputdataId = new TH1F*[T_PCA->GetNbranches()];
    
    TH1F** Cumulative = new TH1F*[T_PCA->GetNbranches()];
    //TH1F** InvertCumulative = new TH1F*[T_PCA->GetNbranches()];
    
    TSpline3** spline = new TSpline3*[T_PCA->GetNbranches()];
    // TF1** func = new TF1*[T_PCA->GetNbranches()];
    
    TH1F** houtdata = new TH1F*[T_PCA->GetNbranches()];
    TH1F** houtdataUniform = new TH1F*[T_PCA->GetNbranches()];
    TH1F** houtdataId = new TH1F*[T_PCA->GetNbranches()];
    TH1F** hPCA = new TH1F*[T_PCA->GetNbranches()];
    
    int tmp = 0;
    for (int k= 0; k<25; k++) {
        
        if (T_Gauss->GetBranch(Form("data_Gauss_%i",k)) != NULL){
            
            hinputdata[tmp] = new TH1F(Form("hinputdata%i",k), Form("hinputdata%i",k), 100,-5,5);
            hinputdataUniform[tmp] = new TH1F(Form("houtdataUniform%i",k), Form("houtdataUniform%i",k), 100,0,1);
            hinputdataId[tmp] = new TH1F(Form("houtdataId%i",k), Form("houtdataId%i",k), 100,0,1);
            
            houtdata[tmp] = new TH1F(Form("houtdata%i",k), Form("houtdata%i",k), 100,-5,5);
            houtdataUniform[tmp] = new TH1F(Form("houtdataUniform%i",k), Form("houtdataUniform%i",k), 100,0,1);
            houtdataId[tmp] = new TH1F(Form("houtdataId%i",k), Form("houtdataId%i",k), 100,0,1);
            hPCA[tmp] = new TH1F(Form("hPCA%i",k), Form("hPCA%i",k), 100,-10,10);
            
            Cumulative[tmp] = (TH1F*)input->Get(Form("hCumulative%i",k)); Cumulative[tmp]->Print();
            // Cumulative[tmp]->Rebin(5);
            
            //  func[tmp] = Cumulative[tmp]->GetFunction(Form("func%i",k));
            
            /* vector<double> Bins;
             for (int i=1; i<=Cumulative[tmp]->GetNbinsX() ; i++) {
             Bins.push_back(Cumulative[tmp]->GetBinContent(i)); std::cout<<Bins[i-1] <<"\t";
             }
             std::cout<<std::endl <<std::endl <<std::endl;
             
             InvertCumulative[tmp] = new TH1F(Form("InvertCumulative%i",k), Form("InvertCumulative%i",k), Bins.size(),&Bins[0]);
             XXX
             for (int i=1; i<=InvertCumulative[tmp]->GetNbinsX(); i++) {
             
             InvertCumulative[tmp]->SetBinContent(i,Cumulative[tmp]->GetBinCenter(i));
             }
             XXX*/
            
            spline[tmp] = new TSpline3(Cumulative[tmp],"Sp",0,1);
            
            tmp = tmp + 1;
        }
    }
    
    //Get PCA Transformation
    
    int Nbranches = T_PCA->GetNbranches(); //std::cout<<Nbranches <<std::endl;
    TMatrixD *EigenVectors = (TMatrixD*)input2->Get("output1_EigenVectors"); //EigenVectors->Print();
    TVectorD *EigenValues = (TVectorD*)input2->Get("output1_EigenValues");
    TVectorD *MeanValues = (TVectorD*)input2->Get("output1_MeanValues");
    TVectorD *SigmaValues = (TVectorD*)input2->Get("output1_SigmaValues");
    
    
    //Get Input PC data and intvert-PCA
    TreeReader* nr_Gauss = new TreeReader;
	nr_Gauss->SetTree(T_Gauss);
	int NEntries = nr_Gauss->GetEntries();
    
    TreeReader* nr_PCA = new TreeReader;
	nr_PCA->SetTree(T_PCA);
    int NEntries_PCA = nr_Gauss->GetEntries();
    
    TreeReader* nr_Id = new TreeReader;
	nr_Id->SetTree(T_Id);
    
    TreeReader* nr_Uniform = new TreeReader;
	nr_Uniform->SetTree(T_Uniform);
    
    
    for ( int ientry=0;ientry<NEntries;ientry++ ){
        
        
        nr_Gauss->GetEntry(ientry);
        nr_PCA->GetEntry(ientry);
        nr_Uniform->GetEntry(ientry);
        nr_Id->GetEntry(ientry);
        
        int tmp2 = 0;
        for (int k= 0; k<25; k++) {
            if (T_Gauss->GetBranch(Form("data_Gauss_%i",k)) != NULL){
                
                hinputdata[tmp2]->Fill(nr_Gauss->GetVariable(Form("data_Gauss_%i",k)));
                
                hinputdataUniform[tmp2]->Fill(nr_Uniform->GetVariable(Form("data_Uniform_%i",k)));
                
                hinputdataId[tmp2]->Fill(nr_Id->GetVariable(Form("data_%i",k)));
                
                hPCA[tmp2]->Fill(nr_PCA->GetVariable(Form("PCA_%i",tmp2)));
                
                tmp2 = tmp2 + 1;
            }
            
        }
    }
    
    //Pull random variable from input Principal Component
    Double_t* PCA_random = new Double_t[T_PCA->GetNbranches()];
    Double_t* outputdata = new Double_t[T_PCA->GetNbranches()];
    
    for (int l=1; l<=T_PCA->GetNbranches(); l++) {
        
        for ( int ientry=0;ientry<NEntries_PCA;ientry++ ){
            nr_PCA->GetEntry(ientry);
            int tmp3 = 0;
            for (int k= 0; k<25; k++) {
                if (T_Gauss->GetBranch(Form("data_Gauss_%i",k)) != NULL){
                    PCA_random[tmp3] = hPCA[tmp3]->GetRandom();
                    tmp3 = tmp3 + 1;
                }
                
            }
            
            //Invert PCA
            P2X(T_PCA->GetNbranches(), EigenVectors,MeanValues, SigmaValues, PCA_random, outputdata , l);
            
            for (int m = 0; m<T_PCA->GetNbranches(); m++) {
                houtdata[m]->Fill(outputdata[m]);
                houtdataUniform[m]->Fill((TMath::Erf(outputdata[m]/1.414213562)+1)/2.);
                //houtdataId[m]->Fill(func[m]->GetX((TMath::Erf(outputdata[m]/1.414213562)+1)/2., 0, 1));
                
                //houtdataId[m]->Fill(spline[tmp]->Eval((TMath::Erf(outputdata[m]/1.414213562)+1)/2.));
            }
            
            
        }
        
        
        //Plotter
        int tmp4 = 0;
        for (int k=0; k<25 ; k++) {
            if (T_Gauss->GetBranch(Form("data_Gauss_%i",k)) != NULL){
                
                std::string outputname = Form("Toy1_data_Gauss_%i_Invert_PCA_with_%iVar",k,l);
                PlotterToy1(hinputdata[tmp4], houtdata[tmp4], outputname);
                
                std::string outputname2 = Form("Toy1_data_Uniform_%i_Invert_PCA_with_%iVar",k,l);
                PlotterToy1(hinputdataUniform[tmp4], houtdataUniform[tmp4], outputname2);
                
                std::string outputname3 = Form("Toy1_data_Id_%i_Invert_PCA_with_%iVar",k,l);
                PlotterToy1(hinputdataId[tmp4], houtdataId[tmp4], outputname3);
                
                
                houtdata[tmp4]->Reset();
                houtdataUniform[tmp4]->Reset();
                houtdataId[tmp4]->Reset();
                
                tmp4 = tmp4 +1;
                
            }
        }
        
        
    }

    
    
    
    


}


////////////////////////////////////////////////////
void ToySimulation2(TFile* input2){
////////////////////////////////////////////////////
    
    TRandom3 r;
    double random = r.Uniform(0,1);
    
    
    //Input Tree Reader
    TTree *T_Id = (TTree*)input->Get("Id");
    TTree *T_Uniform = (TTree*)input->Get("Uniform");
    TTree *T_Gauss = (TTree*)input->Get("Gauss");
    TTree *T_PCA = (TTree*)input->Get("PCA");
    
    // Input/Output Histogram :
    TH1F** hinputdata = new TH1F*[T_PCA->GetNbranches()];
    TH1F** hinputdataUniform = new TH1F*[T_PCA->GetNbranches()];
    TH1F** hinputdataId = new TH1F*[T_PCA->GetNbranches()];
    
    TH1F** Cumulative = new TH1F*[T_PCA->GetNbranches()];
    //TH1F** InvertCumulative = new TH1F*[T_PCA->GetNbranches()];
    
    TSpline3** spline = new TSpline3*[T_PCA->GetNbranches()];
    // TF1** func = new TF1*[T_PCA->GetNbranches()];
    
    TH1F** houtdata = new TH1F*[T_PCA->GetNbranches()];
    TH1F** houtdataUniform = new TH1F*[T_PCA->GetNbranches()];
    TH1F** houtdataId = new TH1F*[T_PCA->GetNbranches()];
    TH1F** hPCA = new TH1F*[T_PCA->GetNbranches()];
    
    int tmp = 0;
    for (int k= 0; k<25; k++) {
        
        if (T_Gauss->GetBranch(Form("data_Gauss_%i",k)) != NULL){
            
            hinputdata[tmp] = new TH1F(Form("hinputdata%i",k), Form("hinputdata%i",k), 100,-5,5);
            hinputdataUniform[tmp] = new TH1F(Form("houtdataUniform%i",k), Form("houtdataUniform%i",k), 100,0,1);
            hinputdataId[tmp] = new TH1F(Form("houtdataId%i",k), Form("houtdataId%i",k), 100,0,1);
            
            houtdata[tmp] = new TH1F(Form("houtdata%i",k), Form("houtdata%i",k), 100,-5,5);
            houtdataUniform[tmp] = new TH1F(Form("houtdataUniform%i",k), Form("houtdataUniform%i",k), 100,0,1);
            houtdataId[tmp] = new TH1F(Form("houtdataId%i",k), Form("houtdataId%i",k), 100,0,1);
            hPCA[tmp] = new TH1F(Form("hPCA%i",k), Form("hPCA%i",k), 100,-10,10);
            
            Cumulative[tmp] = (TH1F*)input->Get(Form("hCumulative%i",k)); Cumulative[tmp]->Print();
            // Cumulative[tmp]->Rebin(5);
            
            //  func[tmp] = Cumulative[tmp]->GetFunction(Form("func%i",k));
            
            /* vector<double> Bins;
             for (int i=1; i<=Cumulative[tmp]->GetNbinsX() ; i++) {
             Bins.push_back(Cumulative[tmp]->GetBinContent(i)); std::cout<<Bins[i-1] <<"\t";
             }
             std::cout<<std::endl <<std::endl <<std::endl;
             
             InvertCumulative[tmp] = new TH1F(Form("InvertCumulative%i",k), Form("InvertCumulative%i",k), Bins.size(),&Bins[0]);
             XXX
             for (int i=1; i<=InvertCumulative[tmp]->GetNbinsX(); i++) {
             
             InvertCumulative[tmp]->SetBinContent(i,Cumulative[tmp]->GetBinCenter(i));
             }
             XXX*/
            
            spline[tmp] = new TSpline3(Cumulative[tmp],"Sp",0,1);
            
            tmp = tmp + 1;
        }
    }
    
    //Get PCA Transformation
    
    int Nbranches = T_PCA->GetNbranches(); //std::cout<<Nbranches <<std::endl;
    TMatrixD *EigenVectors = (TMatrixD*)input2->Get("output1_EigenVectors"); //EigenVectors->Print();
    TVectorD *EigenValues = (TVectorD*)input2->Get("output1_EigenValues");
    TVectorD *MeanValues = (TVectorD*)input2->Get("output1_MeanValues");
    TVectorD *SigmaValues = (TVectorD*)input2->Get("output1_SigmaValues");
    
    //Get Input PC data and intvert-PCA
    TreeReader* nr_Gauss = new TreeReader;
	nr_Gauss->SetTree(T_Gauss);
	int NEntries = nr_Gauss->GetEntries();
    
    TreeReader* nr_PCA = new TreeReader;
	nr_PCA->SetTree(T_PCA);
    int NEntries_PCA = nr_Gauss->GetEntries();
    
    TreeReader* nr_Id = new TreeReader;
	nr_Id->SetTree(T_Id);
    
    TreeReader* nr_Uniform = new TreeReader;
	nr_Uniform->SetTree(T_Uniform);
    
    
    for ( int ientry=0;ientry<NEntries;ientry++ ){
        
        
        nr_Gauss->GetEntry(ientry);
        nr_PCA->GetEntry(ientry);
        nr_Uniform->GetEntry(ientry);
        nr_Id->GetEntry(ientry);
        
        int tmp2 = 0;
        for (int k= 0; k<25; k++) {
            if (T_Gauss->GetBranch(Form("data_Gauss_%i",k)) != NULL){
                
                hinputdata[tmp2]->Fill(nr_Gauss->GetVariable(Form("data_Gauss_%i",k)));
                
                hinputdataUniform[tmp2]->Fill(nr_Uniform->GetVariable(Form("data_Uniform_%i",k)));
                
                hinputdataId[tmp2]->Fill(nr_Id->GetVariable(Form("data_%i",k)));
                
                hPCA[tmp2]->Fill(nr_PCA->GetVariable(Form("PCA_%i",tmp2)));
                
                tmp2 = tmp2 + 1;
            }
            
        }
    }
    
    //Pull random variable from input Principal Component
    Double_t* PCA_random = new Double_t[T_PCA->GetNbranches()];
    Double_t* outputdata = new Double_t[T_PCA->GetNbranches()];
    
    for (int l=1; l<=T_PCA->GetNbranches(); l++) {
        
        for ( int ientry=0;ientry<NEntries_PCA;ientry++ ){
            nr_PCA->GetEntry(ientry);
            int tmp3 = 0;
            for (int k= 0; k<25; k++) {
                if (T_Gauss->GetBranch(Form("data_Gauss_%i",k)) != NULL){
                    PCA_random[tmp3] = hPCA[tmp3]->GetRandom();
                    tmp3 = tmp3 + 1;
                }
                
            }
            
            //Invert PCA
            P2X(T_PCA->GetNbranches(), EigenVectors,MeanValues, SigmaValues, PCA_random, outputdata , l);
            
            for (int m = 0; m<T_PCA->GetNbranches(); m++) {
                houtdata[m]->Fill(outputdata[m]);
                houtdataUniform[m]->Fill((TMath::Erf(outputdata[m]/1.414213562)+1)/2.);
                //houtdataId[m]->Fill(func[m]->GetX((TMath::Erf(outputdata[m]/1.414213562)+1)/2., 0, 1));
                
                //houtdataId[m]->Fill(spline[tmp]->Eval((TMath::Erf(outputdata[m]/1.414213562)+1)/2.));
            }
            
            
        }
        
        
        //Plotter
        int tmp4 = 0;
        for (int k=0; k<25 ; k++) {
            if (T_Gauss->GetBranch(Form("data_Gauss_%i",k)) != NULL){
                
                std::string outputname = Form("Toy1_data_Gauss_%i_Invert_PCA_with_%iVar",k,l);
                PlotterToy1(hinputdata[tmp4], houtdata[tmp4], outputname);
                
                std::string outputname2 = Form("Toy1_data_Uniform_%i_Invert_PCA_with_%iVar",k,l);
                PlotterToy1(hinputdataUniform[tmp4], houtdataUniform[tmp4], outputname2);
                
                std::string outputname3 = Form("Toy1_data_Id_%i_Invert_PCA_with_%iVar",k,l);
                PlotterToy1(hinputdataId[tmp4], houtdataId[tmp4], outputname3);
                
                
                houtdata[tmp4]->Reset();
                houtdataUniform[tmp4]->Reset();
                houtdataId[tmp4]->Reset();
                
                tmp4 = tmp4 +1;
                
            }
        }
        
        
    }
   
    
}


////////////////////////////////////////////////////
void PCA_Validation(TString InputFile="PCAOutput.root", TString InputFile2="PCA_transformation_inputs.root"){
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
    TTree *T_Uniform = (TTree*)input->Get("Uniform");
    TTree *T_Gauss = (TTree*)input->Get("Gauss");
    TTree *T_PCA = (TTree*)input->Get("PCA");

    //--------------------------------------------------------------------------
    //                  Plot transformation chain
    //--------------------------------------------------------------------------

    for (int i=0; i<25; i++) {
            if (T_Id->GetBranch(Form("data_%i",i)) != NULL  && T_Uniform->GetBranch(Form("data_Uniform_%i",i)) != NULL && T_Gauss->GetBranch(Form("data_Gauss_%i",i)) != NULL) {
               
                std::string VarPlotted_Id = Form("data_%i",i);
                std::string VarPlotted_Uniform = Form("data_Uniform_%i",i);
                std::string VarPlotted_Gauss = Form("data_Gauss_%i",i);
                
                TH1F *Cumulative = (TH1F*)input->Get(Form("hCumulative%i",i));
                
                std::string output = Form("TransformatioChain_Var_%i",i);

                TransformationChainPlots(T_Id, VarPlotted_Id, T_Uniform, VarPlotted_Uniform, T_Gauss, VarPlotted_Gauss, Cumulative, output);
            
                delete Cumulative;
        
            }
    }
 
    //--------------------------------------------------------------------------
    //                  Scatter Plots with fits
    //--------------------------------------------------------------------------

    //Scatter Plots Id
    for (int i=0; i<25; i++) {
        for (int j = i; j<25; j++) {
            if (T_Id->GetBranch(Form("data_%i",i)) != NULL  && T_Id->GetBranch(Form("data_%i",j)) != NULL && i != j) {
                std::string VarPlotted = Form("data_%i:data_%i",i,j);
                
                std::string output = Form("data_%i_vs_data_%i",i,j);
                ScatterPlots(T_Id, VarPlotted, output);
            }
        }
        
    }

    
    //Scatter Plots Uniform
    for (int i=0; i<25; i++) {
        for (int j = i; j<25; j++) {
            if (T_Uniform->GetBranch(Form("data_Uniform_%i",i)) != NULL  && T_Uniform->GetBranch(Form("data_Uniform_%i",j)) != NULL  && i != j) {
                std::string VarPlotted = Form("data_Uniform_%i:data_Uniform_%i",i,j);
                
                std::string output = Form("data_Uniform_%i_vs_data_Uniform_%i",i,j);
                ScatterPlots(T_Uniform, VarPlotted, output);
            }
        }
        
    }
    
    //Scatter Plots Gauss
    for (int i=0; i<25; i++) {
        for (int j = i; j<25; j++) {
            if (T_Gauss->GetBranch(Form("data_Gauss_%i",i)) != NULL  && T_Gauss->GetBranch(Form("data_Gauss_%i",j)) != NULL  && i != j) {
                std::string VarPlotted = Form("data_Gauss_%i:data_Gauss_%i",i,j);
                
                std::string output = Form("data_Gauss_%i_vs_data_Gauss_%i",i,j);
                ScatterPlots(T_Gauss, VarPlotted, output);
            }
        }
        
    }
    
    //Scatter Plots PCA
    for (int i=0; i<25; i++) {
        for (int j = i; j<25; j++) {
            if (T_PCA->GetBranch(Form("PCA_%i",i)) != NULL  && T_PCA->GetBranch(Form("PCA_%i",j)) != NULL  && i != j) {
                std::string VarPlotted = Form("PCA_%i:PCA_%i",i,j);
                
                std::string output = Form("PCA_%i_vs_PCA_%i",i,j);
                ScatterPlots(T_PCA, VarPlotted, output);
            }
        }
        
    }
    
    //--------------------------------------------------------------------------
    //                  Toy Simulation Validation
    //--------------------------------------------------------------------------
    
    ToySimulation1(input, input2);
    
    
    
    
    

    
    
}


















