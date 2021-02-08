/*
  Copyright (C) 2002-2017 CERN for the benefit of the ATLAS collaboration
*/

#include "TSystem.h"
#include "TROOT.h"
#include "THtml.h"

#include <iostream>

void initTFCSShapeValidation() {

    // * load required macros

    std::cout << " ** including paths and loading macros ...." << std::endl;
    std::string AthenaPath = "../../athena/Simulation/ISF/ISF_FastCaloSim";
    std::string EnergyParameterizationPath = "../../EnergyParametrization";


    gInterpreter->AddIncludePath("../");
    gInterpreter->AddIncludePath("../FastCaloSimAnalyzer/");
    gInterpreter->AddIncludePath((EnergyParameterizationPath + "/src/").c_str());
    gInterpreter->AddIncludePath((AthenaPath + "/ISF_FastCaloSimEvent/").c_str());
    gInterpreter->AddIncludePath((AthenaPath + "ISF_FastCaloSimParametrization/").c_str());
    gInterpreter->AddIncludePath((AthenaPath + "/ISF_FastCaloSimParametrization/tools/").c_str());

    /*
    TString compile_cmd=gSystem->GetMakeSharedLib();
    compile_cmd.ReplaceAll("$SourceFiles","-D__FastCaloSimStandAlone__ \"$SourceFiles\"");
    gSystem->SetMakeSharedLib(compile_cmd);
    */

    gEnv->SetValue("ACLiC.IncludePaths", TString(gEnv->GetValue("ACLiC.IncludePaths", "")) + " -D__FastCaloSimStandAlone__ ");

    gROOT->LoadMacro("../atlasstyle/AtlasStyle.C+");
    gROOT->LoadMacro("../atlasstyle/AtlasLabels.C+");
    gROOT->LoadMacro("../atlasstyle/AtlasUtils.C+");


    gROOT->LoadMacro("../CLHEP/Random/RandomEngine.cxx+");
    gROOT->LoadMacro("../CLHEP/Random/TRandomEngine.cxx+");
    gROOT->LoadMacro("../CLHEP/Random/RandFlat.cxx+");
    gROOT->LoadMacro("../CLHEP/Random/RandGauss.cxx+");
    gROOT->LoadMacro("../CLHEP/Random/RandPoisson.cxx+");

    gROOT->LoadMacro((AthenaPath + "/ISF_FastCaloSimParametrization/tools/CaloSampling.cxx+").c_str());
    gROOT->LoadMacro((AthenaPath + "/ISF_FastCaloSimEvent/src/TFCSTruthState.cxx+").c_str());
    gROOT->LoadMacro((AthenaPath + "/ISF_FastCaloSimEvent/src/TFCSExtrapolationState.cxx+").c_str());
    gROOT->LoadMacro((AthenaPath + "/ISF_FastCaloSimEvent/src/TFCSSimulationState.cxx+").c_str());
    gROOT->LoadMacro((AthenaPath + "/ISF_FastCaloSimEvent/src/TFCSParametrizationBase.cxx+").c_str());
    gROOT->LoadMacro((AthenaPath + "/ISF_FastCaloSimEvent/src/TFCSParametrization.cxx+").c_str());
    gROOT->LoadMacro((AthenaPath + "/ISF_FastCaloSimEvent/src/TFCSInvisibleParametrization.cxx+").c_str());
    gROOT->LoadMacro((AthenaPath + "/ISF_FastCaloSimEvent/src/TFCSInitWithEkin.cxx+").c_str());
    gROOT->LoadMacro((AthenaPath + "/ISF_FastCaloSimEvent/src/TFCSEnergyInterpolationLinear.cxx+").c_str());
    gROOT->LoadMacro((AthenaPath + "/ISF_FastCaloSimEvent/src/TFCSEnergyInterpolationSpline.cxx+").c_str());
    gROOT->LoadMacro((AthenaPath + "/ISF_FastCaloSimEvent/src/TFCSParametrizationChain.cxx+").c_str());
    gROOT->LoadMacro((AthenaPath + "/ISF_FastCaloSimEvent/src/TFCSEnergyParametrization.cxx+").c_str());
    gROOT->LoadMacro((AthenaPath + "/ISF_FastCaloSimEvent/src/TFCSLateralShapeParametrization.cxx+").c_str());
    gROOT->LoadMacro((AthenaPath + "/ISF_FastCaloSimEvent/src/TFCSParametrizationBinnedChain.cxx+").c_str());
    gROOT->LoadMacro((AthenaPath + "/ISF_FastCaloSimEvent/src/TFCSParametrizationEbinChain.cxx+").c_str());
    gROOT->LoadMacro((AthenaPath + "/ISF_FastCaloSimEvent/src/TFCSParametrizationPDGIDSelectChain.cxx+").c_str());
    gROOT->LoadMacro((AthenaPath + "/ISF_FastCaloSimEvent/src/TFCSParametrizationFloatSelectChain.cxx+").c_str());
    gROOT->LoadMacro((AthenaPath + "/ISF_FastCaloSimEvent/src/TFCSParametrizationEkinSelectChain.cxx+").c_str());
    gROOT->LoadMacro((AthenaPath + "/ISF_FastCaloSimEvent/src/TFCSParametrizationEtaSelectChain.cxx+").c_str());
    gROOT->LoadMacro((AthenaPath + "/ISF_FastCaloSimEvent/src/TFCSParametrizationAbsEtaSelectChain.cxx+").c_str());

    gROOT->LoadMacro((AthenaPath + "/ISF_FastCaloSimEvent/src/TFCSFunction.cxx+").c_str());
    gROOT->LoadMacro((AthenaPath + "/ISF_FastCaloSimEvent/src/TFCS1DFunction.cxx+").c_str());
    gROOT->LoadMacro((AthenaPath + "/ISF_FastCaloSimEvent/src/TFCS1DFunctionHistogram.cxx+").c_str());
    gROOT->LoadMacro((AthenaPath + "/ISF_FastCaloSimEvent/src/TFCS1DFunctionInt16Histogram.cxx+").c_str());
    gROOT->LoadMacro((AthenaPath + "/ISF_FastCaloSimEvent/src/TFCS1DFunctionInt32Histogram.cxx+").c_str());
    gROOT->LoadMacro((AthenaPath + "/ISF_FastCaloSimEvent/src/TFCS1DFunctionRegression.cxx+").c_str());
    gROOT->LoadMacro((AthenaPath + "/ISF_FastCaloSimEvent/src/TFCS1DFunctionSpline.cxx+").c_str());


    //gDebug=7;
    gROOT->LoadMacro((AthenaPath + "/ISF_FastCaloSimEvent/src/TFCS1DFunctionTemplateHistogram.cxx+").c_str());
    //gDebug=0;
    gROOT->LoadMacro((AthenaPath + "/ISF_FastCaloSimEvent/src/TFCS2DFunction.cxx+").c_str());
    gROOT->LoadMacro((AthenaPath + "/ISF_FastCaloSimEvent/src/TFCS2DFunctionHistogram.cxx+").c_str());

    gROOT->LoadMacro((AthenaPath + "/ISF_FastCaloSimEvent/src/IntArray.cxx+").c_str());
    gROOT->LoadMacro((AthenaPath + "/ISF_FastCaloSimEvent/src/DoubleArray.cxx+").c_str());
    gROOT->LoadMacro((AthenaPath + "/ISF_FastCaloSimEvent/src/TFCSPCAEnergyParametrization.cxx+").c_str());
    gROOT->LoadMacro((AthenaPath + "/ISF_FastCaloSimEvent/src/TFCSEnergyBinParametrization.cxx+").c_str());

    gROOT->LoadMacro((AthenaPath + "/ISF_FastCaloSimEvent/src/TFCSLateralShapeParametrizationHitBase.cxx+").c_str());
    gROOT->LoadMacro((AthenaPath + "/ISF_FastCaloSimEvent/src/TFCSLateralShapeParametrizationHitChain.cxx+").c_str());
    gROOT->LoadMacro((AthenaPath + "/ISF_FastCaloSimEvent/src/TFCSHistoLateralShapeParametrization.cxx+").c_str());
    gROOT->LoadMacro((AthenaPath + "/ISF_FastCaloSimEvent/src/TFCSLateralShapeParametrizationHitNumberFromE.cxx+").c_str());
    gROOT->LoadMacro((AthenaPath + "/ISF_FastCaloSimEvent/src/TFCSHitCellMapping.cxx+").c_str());
    gROOT->LoadMacro((AthenaPath + "/ISF_FastCaloSimEvent/src/TFCSHitCellMappingFCal.cxx+").c_str());

    gROOT->LoadMacro((AthenaPath + "/ISF_FastCaloSimEvent/src/TFCSHitCellMappingWiggle.cxx+").c_str());
    gROOT->LoadMacro((AthenaPath + "/ISF_FastCaloSimEvent/src/TFCSHitCellMappingWiggleEMB.cxx+").c_str());

    gROOT->LoadMacro((AthenaPath + "/ISF_FastCaloSimParametrization/Root/TFCSSimpleLateralShapeParametrization.cxx+").c_str());
    gROOT->LoadMacro((AthenaPath + "/ISF_FastCaloSimParametrization/Root/TFCSNNLateralShapeParametrization.cxx+").c_str());
    gROOT->LoadMacro((AthenaPath + "/ISF_FastCaloSimParametrization/src/CaloGeometry.cxx+").c_str());
    gROOT->LoadMacro((AthenaPath + "/ISF_FastCaloSimParametrization/src/CaloGeometryLookup.cxx+").c_str());

    gROOT->LoadMacro((AthenaPath + "/ISF_FastCaloSimParametrization/tools/CaloGeometryFromFile.cxx+").c_str());
    gROOT->LoadMacro((AthenaPath + "/ISF_FastCaloSimParametrization/tools/FCAL_ChannelMap.cxx+").c_str());

//    gROOT->LoadMacro("../Root/TreeReader.cxx+");
    gROOT->LoadMacro("../Root/TFCSAnalyzerBase.cxx+");
    gROOT->LoadMacro((EnergyParameterizationPath + "/src/TreeReader.cxx+").c_str());
    gROOT->LoadMacro((EnergyParameterizationPath + "/src/TFCSMakeFirstPCA.cxx+").c_str());
    gROOT->LoadMacro((EnergyParameterizationPath + "/src/TFCSApplyFirstPCA.cxx+").c_str());
//    gROOT->LoadMacro("../Root/TFCSfirstPCA.cxx+");
//    gROOT->LoadMacro("../Root/TFCSFlatNtupleMaker.cxx+");
    gROOT->LoadMacro("../Root/TFCSValidationEnergy.cxx+");
    gROOT->LoadMacro("../Root/TFCSValidationEnergyAndCells.cxx+");
    gROOT->LoadMacro("../Root/TFCSValidationHitSpy.cxx+");
    gROOT->LoadMacro("../Root/TFCSValidationEnergyAndHits.cxx+");
    gROOT->LoadMacro("../Root/TFCSShapeValidation.cxx+");
    gROOT->LoadMacro("../Root/TFCSWriteCellsToTree.cxx+");
//    gROOT->LoadMacro("../Root/TFCSInputValidationPlots.cxx+");

    gROOT->LoadMacro("FCS_dsid.cxx+");

    if (1 == 0) {
        THtml html;
        html.SetProductName("TFCS");
        html.SetInputDir((AthenaPath + "/ISF_FastCaloSimParametrization/:" + AthenaPath + "/ISF_FastCaloSimParametrization/tools/:" + AthenaPath + "/ISF_FastCaloSimEvent/:../FastCaloSimAnalyzer/").c_str());
        html.MakeAll();
    }
}
