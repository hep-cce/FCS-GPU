/*
  Copyright (C) 2002-2017 CERN for the benefit of the ATLAS collaboration
*/

{
  gStyle->SetOptStat(0);

  gSystem->AddIncludePath(" -I.. ");

  gROOT->LoadMacro("../ISF_FastCaloSimParametrization/MeanAndRMS.h+");
  gROOT->LoadMacro("Identifier/Identifier.h+");
  gROOT->LoadMacro("CaloDetDescr/CaloDetDescrElement.h+");
  gROOT->LoadMacro("CaloSampling.cxx+");
  gROOT->LoadMacro("../src/FCAL_ChannelMap.cxx+");
  gROOT->LoadMacro("../src/CaloGeometry.cxx+");
  gROOT->LoadMacro("CaloGeometryFromFile.cxx+");
  
  gROOT->LoadMacro("wiggle_closure_inputs.cxx+");
  //wiggle_closure_inputs("Sampling_0");
  //wiggle_closure_inputs("Sampling_1");
  //wiggle_closure_inputs("Sampling_2");
  //wiggle_closure_inputs("Sampling_3");
  //wiggle_closure_inputs("Sampling_12");
  //wiggle_closure_inputs("Sampling_13");
	//wiggle_closure_inputs("Sampling_14");
		wiggle_closure_inputs("Sampling_21");
}
