/*
  Copyright (C) 2002-2017 CERN for the benefit of the ATLAS collaboration
*/

//void run(TString sampling="Sampling_0"){
{
  gStyle->SetOptStat(0);
  gStyle->SetPalette(1);

  gSystem->AddIncludePath(" -I.. ");

  gROOT->LoadMacro("../ISF_FastCaloSimParametrization/MeanAndRMS.h+");
  gROOT->LoadMacro("Identifier/Identifier.h+");
  gROOT->LoadMacro("CaloDetDescr/CaloDetDescrElement.h+");
  gROOT->LoadMacro("CaloSampling.cxx+");
  gROOT->LoadMacro("../src/FCAL_ChannelMap.cxx+");
  gROOT->LoadMacro("../src/CaloGeometry.cxx+");
  gROOT->LoadMacro("CaloGeometryFromFile.cxx+");
    
  gROOT->LoadMacro("wiggleClosureAndComparison.cxx+");
 
  wiggleClosureAndComparison("Sampling_1");
 

}
