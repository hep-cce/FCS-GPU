/*
  Copyright (C) 2002-2017 CERN for the benefit of the ATLAS collaboration
*/

#include "TSystem.h"
#include "TROOT.h"

#include <iostream>

void initCaloGeometry();

void initCaloGeometry()
{

    std::cout << " ** including paths and macros for CaloGeometry tool ...." << std::endl;

    gInterpreter->AddIncludePath("../../ISF_FastCaloSimEvent/");
    gInterpreter->AddIncludePath("../../ISF_FastCaloSimParametrization/");
    gInterpreter->AddIncludePath("../../ISF_FastCaloSimParametrization/tools/");



    gROOT->LoadMacro("../../ISF_FastCaloSimParametrization/src/CaloGeometry.cxx+");
    gROOT->LoadMacro("../../ISF_FastCaloSimParametrization/src/CaloGeometryLookup.cxx+");
    gROOT->LoadMacro("../../ISF_FastCaloSimParametrization/tools/CaloGeometryFromFile.cxx+");
    gROOT->LoadMacro("../../ISF_FastCaloSimParametrization/tools/FCAL_ChannelMap.cxx+");


}
