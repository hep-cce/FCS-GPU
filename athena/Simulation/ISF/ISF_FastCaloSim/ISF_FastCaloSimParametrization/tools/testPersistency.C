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

    gROOT->LoadMacro("../ISF_FastCaloSimParametrization/MeanAndRMS.h+");
    gROOT->LoadMacro("../Root/IntArray.cxx+");
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
/*
    gROOT->LoadMacro("../Root/TFCSLateralShapeParametrization.cxx+");
    gROOT->LoadMacro("../Root/TFCSNNLateralShapeParametrization.cxx+");
    gROOT->LoadMacro("../Root/TFCSSimpleLateralShapeParametrization.cxx+");
*/
  }
  
  //test the TFCSFunction:
  
  //that one works well:
  //TFile *input=TFile::Open("PCA2_bin0_pions_new_2dbin_10bins.root");
  //TH1D* hist=(TH1D*)input->Get("h_cumulative_total")); hist->SetName("hist");
  
  //that one crashes:
  TFile *input=TFile::Open("PCA2_bin8_pions_calo_10bins.root");
  TH1D* hist=(TH1D*)input->Get("h_cumulative_PCA_3")); hist->SetName("hist");
  
  TFCS1DFunction* fct=TFCSFunction::Create(hist,0);
  cout<<fct<<endl;
  
}

