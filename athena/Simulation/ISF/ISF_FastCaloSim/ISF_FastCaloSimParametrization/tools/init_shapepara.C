/*
  Copyright (C) 2002-2017 CERN for the benefit of the ATLAS collaboration
*/

#include "TSystem.h"
#include "TROOT.h"

#include <iostream>

using namespace std;

void init_shapepara(bool);

void init_shapepara(bool isStandAlone) {


    if (!isStandAlone) {
        // * load required macros

        cout << " ** including paths and loading macros ...." << endl;

        gInterpreter->AddIncludePath("..");
        gInterpreter->AddIncludePath("../shapepara");
        gInterpreter->AddIncludePath("../../ISF_FastCaloSimEvent");
        gInterpreter->AddIncludePath("./");


        // gROOT->LoadMacro("CaloSampling.cxx+");
        gROOT->LoadMacro("../src/CaloGeoGeometry.cxx+");
        gROOT->LoadMacro("../src/CaloGeometryFromFile.cxx+");
        gROOT->LoadMacro("../src/FCAL_ChannelMap.cxx+");
        gROOT->LoadMacro("../shapepara/ShowerShapeThinning.cxx+");
        gROOT->LoadMacro("../shapepara/ShowerShapeStudy.cxx+");
        gROOT->LoadMacro("../shapepara/ShowerShapeBinning.cxx+");
        gROOT->LoadMacro("../shapepara/ShowerShapePlotting.cxx+");
        gROOT->LoadMacro("../shapepara/ShowerShapeRegression.cxx+");

    } else if (isStandAlone) {


        cout << " ** configuring for standalone running ...." << endl;

        gInterpreter->AddIncludePath("..");
        gInterpreter->AddIncludePath("../shapepara");
        gInterpreter->AddIncludePath("./");


        gROOT->LoadMacro("../shapepara/ShowerShapeStudy.cxx+");
        gROOT->LoadMacro("../shapepara/ShowerShapeBinning.cxx+");
        gROOT->LoadMacro("../shapepara/ShowerShapePlotting.cxx+");
        gROOT->LoadMacro("../shapepara/ShowerShapeRegression.cxx+");


    }

}
