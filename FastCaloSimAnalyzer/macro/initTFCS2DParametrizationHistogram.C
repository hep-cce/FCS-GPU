/*
  Copyright (C) 2002-2017 CERN for the benefit of the ATLAS collaboration
*/

#include "TSystem.h"
#include "TEnv.h"
#include "TROOT.h"
#include "THtml.h"

#include <iostream>

void initTFCS2DParametrizationHistogram(std::string PATH = "../../") {

    // * load required macros

    std::string EnergyParameterizationPath = "../../EnergyParametrization";

    std::cout << " ** including paths and loading macros ...." << std::endl;

    gInterpreter->AddIncludePath("../");
    gInterpreter->AddIncludePath("../FastCaloSimAnalyzer/");
    gInterpreter->AddIncludePath((EnergyParameterizationPath + "/src/").c_str());
    gInterpreter->AddIncludePath((PATH + "ISF_FastCaloSimEvent/").c_str());
    gInterpreter->AddIncludePath((PATH + "ISF_FastCaloSimParametrization/").c_str());
    gInterpreter->AddIncludePath((PATH + "ISF_FastCaloSimParametrization/tools/").c_str());



    gEnv->SetValue("ACLiC.IncludePaths", TString(gEnv->GetValue("ACLiC.IncludePaths", "")) + " -D__FastCaloSimStandAlone__ ");

    gROOT->LoadMacro("../atlasstyle/AtlasStyle.C+");
    gROOT->LoadMacro("../atlasstyle/AtlasLabels.C+");
    gROOT->LoadMacro("../atlasstyle/AtlasUtils.C+");

    gROOT->LoadMacro("../CLHEP/Random/RandomEngine.cxx+");
    gROOT->LoadMacro("../CLHEP/Random/TRandomEngine.cxx+");
    gROOT->LoadMacro("../CLHEP/Random/RandFlat.cxx+");
    gROOT->LoadMacro("../CLHEP/Random/RandGauss.cxx+");
    gROOT->LoadMacro("../CLHEP/Random/RandPoisson.cxx+");

    gROOT->LoadMacro((PATH + "ISF_FastCaloSimParametrization/tools/CaloSampling.cxx+").c_str());
    gROOT->LoadMacro((PATH + "ISF_FastCaloSimEvent/src/TFCSTruthState.cxx+").c_str());
    gROOT->LoadMacro((PATH + "ISF_FastCaloSimEvent/src/TFCSExtrapolationState.cxx+").c_str());
    gROOT->LoadMacro((PATH + "ISF_FastCaloSimEvent/src/TFCSSimulationState.cxx+").c_str());
    gROOT->LoadMacro((PATH + "ISF_FastCaloSimEvent/src/TFCSParametrizationBase.cxx+").c_str());
    gROOT->LoadMacro((PATH + "ISF_FastCaloSimEvent/src/TFCSParametrization.cxx+").c_str());
    gROOT->LoadMacro((PATH + "ISF_FastCaloSimEvent/src/TFCSInitWithEkin.cxx+").c_str());
    gROOT->LoadMacro((PATH + "ISF_FastCaloSimEvent/src/TFCSParametrizationChain.cxx+").c_str());
    gROOT->LoadMacro((PATH + "ISF_FastCaloSimEvent/src/TFCSEnergyParametrization.cxx+").c_str());
    gROOT->LoadMacro((PATH + "ISF_FastCaloSimEvent/src/TFCSLateralShapeParametrization.cxx+").c_str());
    gROOT->LoadMacro((PATH + "ISF_FastCaloSimEvent/src/TFCSParametrizationBinnedChain.cxx+").c_str());
    gROOT->LoadMacro((PATH + "ISF_FastCaloSimEvent/src/TFCSParametrizationEbinChain.cxx+").c_str());
    gROOT->LoadMacro((PATH + "ISF_FastCaloSimEvent/src/TFCSInvisibleParametrization.cxx+").c_str());
    gROOT->LoadMacro((PATH + "ISF_FastCaloSimEvent/src/TFCSParametrizationPDGIDSelectChain.cxx+").c_str());
    gROOT->LoadMacro((PATH + "ISF_FastCaloSimEvent/src/TFCSParametrizationFloatSelectChain.cxx+").c_str());
    gROOT->LoadMacro((PATH + "ISF_FastCaloSimEvent/src/TFCSParametrizationEkinSelectChain.cxx+").c_str());
    gROOT->LoadMacro((PATH + "ISF_FastCaloSimEvent/src/TFCSParametrizationEtaSelectChain.cxx+").c_str());
    gROOT->LoadMacro((PATH + "ISF_FastCaloSimEvent/src/TFCSParametrizationAbsEtaSelectChain.cxx+").c_str());
    gROOT->LoadMacro((PATH + "ISF_FastCaloSimEvent/src/TFCSEnergyBinParametrization.cxx+").c_str());

    gROOT->LoadMacro((PATH + "ISF_FastCaloSimEvent/src/TFCSFunction.cxx+").c_str());
    gROOT->LoadMacro((PATH + "ISF_FastCaloSimEvent/src/TFCS1DFunction.cxx+").c_str());
    gROOT->LoadMacro((PATH + "ISF_FastCaloSimEvent/src/TFCS2DFunction.cxx+").c_str());
    gROOT->LoadMacro((PATH + "ISF_FastCaloSimEvent/src/TFCS1DFunctionHistogram.cxx+").c_str());
    gROOT->LoadMacro((PATH + "ISF_FastCaloSimEvent/src/TFCS2DFunctionHistogram.cxx+").c_str());
    gROOT->LoadMacro((PATH + "ISF_FastCaloSimEvent/src/TFCS1DFunctionRegression.cxx+").c_str());
    gROOT->LoadMacro((PATH + "ISF_FastCaloSimEvent/src/IntArray.cxx+g").c_str());
    gROOT->LoadMacro((PATH + "ISF_FastCaloSimEvent/src/DoubleArray.cxx+g").c_str());
    gROOT->LoadMacro((PATH + "ISF_FastCaloSimEvent/src/TFCSPCAEnergyParametrization.cxx+g").c_str());
    gROOT->LoadMacro((PATH + "ISF_FastCaloSimEvent/src/TFCSEnergyBinParametrization.cxx+").c_str());

    gROOT->LoadMacro((PATH + "ISF_FastCaloSimEvent/src/TFCSLateralShapeParametrizationHitBase.cxx+").c_str());
    gROOT->LoadMacro((PATH + "ISF_FastCaloSimEvent/src/TFCSLateralShapeParametrizationHitChain.cxx+").c_str());
    gROOT->LoadMacro((PATH + "ISF_FastCaloSimEvent/src/TFCSHistoLateralShapeParametrization.cxx+").c_str());
    gROOT->LoadMacro((PATH + "ISF_FastCaloSimEvent/src/TFCSLateralShapeParametrizationHitNumberFromE.cxx+").c_str());
    gROOT->LoadMacro((PATH + "ISF_FastCaloSimEvent/src/TFCSHitCellMapping.cxx+").c_str());
    gROOT->LoadMacro((PATH + "ISF_FastCaloSimEvent/src/TFCSHitCellMappingFCal.cxx+").c_str());
    gROOT->LoadMacro((PATH + "ISF_FastCaloSimEvent/src/TFCSHitCellMappingWiggleEMB.cxx+").c_str());

    gROOT->LoadMacro((PATH + "ISF_FastCaloSimParametrization/Root/TFCSSimpleLateralShapeParametrization.cxx+").c_str());
    gROOT->LoadMacro((PATH + "ISF_FastCaloSimParametrization/Root/TFCSNNLateralShapeParametrization.cxx+").c_str());
    gROOT->LoadMacro((PATH + "ISF_FastCaloSimParametrization/src/CaloGeometry.cxx+").c_str());
    gROOT->LoadMacro((PATH + "ISF_FastCaloSimParametrization/src/CaloGeometryLookup.cxx+").c_str());

    gROOT->LoadMacro((PATH + "ISF_FastCaloSimParametrization/tools/CaloGeometryFromFile.cxx+").c_str());
    gROOT->LoadMacro((PATH + "ISF_FastCaloSimParametrization/tools/FCAL_ChannelMap.cxx+").c_str());

    gROOT->LoadMacro("../Root/TFCSAnalyzerBase.cxx+");

    gROOT->LoadMacro((EnergyParameterizationPath + "/src/TreeReader.cxx+").c_str());
    gROOT->LoadMacro((EnergyParameterizationPath + "/src/TFCSMakeFirstPCA.cxx+").c_str());
    gROOT->LoadMacro((EnergyParameterizationPath + "/src/TFCSApplyFirstPCA.cxx+").c_str());

    gROOT->LoadMacro("../Root/TFCSValidationEnergy.cxx+g");
    gROOT->LoadMacro("../Root/TFCSValidationEnergyAndCells.cxx+g");
    gROOT->LoadMacro("../Root/TFCSValidationHitSpy.cxx+g");
    gROOT->LoadMacro("../Root/TFCSValidationEnergyAndHits.cxx+g");
    gROOT->LoadMacro("../Root/TFCSShapeValidation.cxx+g");

    if (1 == 0) {
        THtml html;
        html.SetProductName("TFCS");
        html.SetInputDir((PATH + "ISF_FastCaloSimParametrization/:" + PATH + "ISF_FastCaloSimParametrization/tools/:" + PATH + "ISF_FastCaloSimEvent/:../FastCaloSimAnalyzer/").c_str());
        html.MakeAll();
    }
}
