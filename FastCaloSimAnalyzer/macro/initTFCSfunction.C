/*
  Copyright (C) 2002-2017 CERN for the benefit of the ATLAS collaboration
*/

#include "TSystem.h"
#include "TROOT.h"
#include "THtml.h"

#include <iostream>

void initTFCSfunction() {

    // * load required macros

    std::cout << " ** including paths and loading macros ...." << std::endl;

    gInterpreter->AddIncludePath("../");
    gInterpreter->AddIncludePath("../FastCaloSimAnalyzer/");
    gInterpreter->AddIncludePath("../../ISF_FastCaloSimEvent/");
    gInterpreter->AddIncludePath("../../ISF_FastCaloSimParametrization/");
    gInterpreter->AddIncludePath("../../ISF_FastCaloSimParametrization/tools/");
    
    gEnv->SetValue("ACLiC.IncludePaths",TString(gEnv->GetValue("ACLiC.IncludePaths",""))+" -D__FastCaloSimStandAlone__ ");

    gROOT->LoadMacro("../atlasstyle/AtlasStyle.C+");
    gROOT->LoadMacro("../atlasstyle/AtlasLabels.C+");
    gROOT->LoadMacro("../atlasstyle/AtlasUtils.C+");

    gROOT->LoadMacro("../../ISF_FastCaloSimEvent/src/TFCSFunction.cxx+");

    gROOT->LoadMacro("../../ISF_FastCaloSimEvent/src/TFCS1DFunction.cxx+");
    gROOT->LoadMacro("../../ISF_FastCaloSimEvent/src/TFCS1DFunctionHistogram.cxx+");
    gROOT->LoadMacro("../../ISF_FastCaloSimEvent/src/TFCS1DFunctionRegression.cxx+");
    gROOT->LoadMacro("../../ISF_FastCaloSimEvent/src/TFCS1DFunctionInt32Histogram.cxx+");
    gROOT->LoadMacro("../../ISF_FastCaloSimEvent/src/TFCS1DFunctionSpline.cxx+");

    gROOT->LoadMacro("../../ISF_FastCaloSimEvent/src/TFCS2DFunction.cxx+");
    gROOT->LoadMacro("../../ISF_FastCaloSimEvent/src/TFCS2DFunctionHistogram.cxx+");
}
