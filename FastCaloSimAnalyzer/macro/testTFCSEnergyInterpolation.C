/*
  Copyright (C) 2002-2017 CERN for the benefit of the ATLAS collaboration
*/

#include "TSystem.h"
#include "TROOT.h"
#include "THtml.h"

#include <iostream>

void testTFCSEnergyInterpolation() {

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

    gROOT->LoadMacro("../../ISF_FastCaloSimParametrization/tools/CaloSampling.cxx+");
    gROOT->LoadMacro("../../ISF_FastCaloSimEvent/src/TFCSTruthState.cxx+");
    gROOT->LoadMacro("../../ISF_FastCaloSimEvent/src/TFCSExtrapolationState.cxx+");
    gROOT->LoadMacro("../../ISF_FastCaloSimEvent/src/TFCSSimulationState.cxx+");
    gROOT->LoadMacro("../../ISF_FastCaloSimEvent/src/TFCSParametrizationBase.cxx+");
    gROOT->LoadMacro("../../ISF_FastCaloSimEvent/src/TFCSParametrization.cxx+");

    gROOT->LoadMacro("../Root/TFCSEnergyInterpolation.cxx+g");
}
