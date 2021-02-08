/*
  Copyright (C) 2002-2019 CERN for the benefit of the ATLAS collaboration
*/

#include "TSystem.h"
#include "TEnv.h"
#include "TROOT.h"
#include "THtml.h"

#include <iostream>

void initTFCSAnalyzer(std::string PATH = "../../athena/Simulation/ISF/ISF_FastCaloSim/")
{
    std::string FastCaloSimCommonPath = "../FastCaloSimCommon";
    std::string EnergyParameterizationPath = "../EnergyParametrization";
    std::string FastCaloSimAnalyzerPath = "../";

    // * load required macros
    std::cout << " ** including paths and loading macros ...." << std::endl;

    gInterpreter->AddIncludePath( FastCaloSimCommonPath.c_str() );
    gInterpreter->AddIncludePath( (FastCaloSimCommonPath + "/src/").c_str() );
    gInterpreter->AddIncludePath( FastCaloSimAnalyzerPath.c_str() );
    gInterpreter->AddIncludePath( (FastCaloSimAnalyzerPath + "/FastCaloSimAnalyzer/").c_str() );

    gInterpreter->AddIncludePath((EnergyParameterizationPath + "/src/").c_str());
    gInterpreter->AddIncludePath((PATH + "ISF_FastCaloSimEvent/").c_str() );
    gInterpreter->AddIncludePath((PATH + "ISF_FastCaloSimParametrization/").c_str() );
    gInterpreter->AddIncludePath((PATH + "ISF_FastCaloSimParametrization/tools/").c_str() );



    gEnv->SetValue("ACLiC.IncludePaths", TString(gEnv->GetValue("ACLiC.IncludePaths", "")) + " -D__FastCaloSimStandAlone__ ");



    gROOT->LoadMacro( (FastCaloSimCommonPath + "/dependencies/atlasrootstyle/AtlasStyle.C+").c_str() );
    gROOT->LoadMacro( (FastCaloSimCommonPath + "/dependencies/atlasrootstyle/AtlasLabels.C+").c_str() );
    gROOT->LoadMacro( (FastCaloSimCommonPath + "/dependencies/atlasrootstyle/AtlasUtils.C+").c_str() );

    gROOT->LoadMacro( (FastCaloSimCommonPath + "/HepPDT/ParticleID.cxx+").c_str() );

    gROOT->LoadMacro( (FastCaloSimCommonPath + "/CLHEP/Random/RandomEngine.cxx+").c_str() );
    gROOT->LoadMacro( (FastCaloSimCommonPath + "/CLHEP/Random/TRandomEngine.cxx+").c_str() );
    gROOT->LoadMacro( (FastCaloSimCommonPath + "/CLHEP/Random/RandFlat.cxx+").c_str() );
    gROOT->LoadMacro( (FastCaloSimCommonPath + "/CLHEP/Random/RandGauss.cxx+").c_str() );
    gROOT->LoadMacro( (FastCaloSimCommonPath + "/CLHEP/Random/RandPoisson.cxx+").c_str() );

    gROOT->LoadMacro((PATH + "ISF_FastCaloSimParametrization/tools/CaloSampling.cxx+").c_str());
    gROOT->LoadMacro((PATH + "ISF_FastCaloSimEvent/src/TFCSTruthState.cxx+").c_str());
    gROOT->LoadMacro((PATH + "ISF_FastCaloSimEvent/src/TFCSExtrapolationState.cxx+").c_str());
    gROOT->LoadMacro((PATH + "ISF_FastCaloSimEvent/src/TFCSSimulationState.cxx+").c_str());
    gROOT->LoadMacro((PATH + "ISF_FastCaloSimEvent/src/TFCSParametrizationBase.cxx+").c_str());
    gROOT->LoadMacro((PATH + "ISF_FastCaloSimEvent/src/TFCSParametrizationPlaceholder.cxx+").c_str());
    gROOT->LoadMacro((PATH + "ISF_FastCaloSimEvent/src/TFCSParametrization.cxx+").c_str());
    gROOT->LoadMacro((PATH + "ISF_FastCaloSimEvent/src/TFCSInvisibleParametrization.cxx+").c_str());
    gROOT->LoadMacro((PATH + "ISF_FastCaloSimEvent/src/TFCSInitWithEkin.cxx+").c_str());
    gROOT->LoadMacro((PATH + "ISF_FastCaloSimEvent/src/TFCSEnergyInterpolationLinear.cxx+").c_str());
    gROOT->LoadMacro((PATH + "ISF_FastCaloSimEvent/src/TFCSEnergyInterpolationSpline.cxx+").c_str());

    gROOT->LoadMacro((PATH + "ISF_FastCaloSimEvent/src/TFCSParametrizationChain.cxx+").c_str());
    gROOT->LoadMacro((PATH + "ISF_FastCaloSimEvent/src/TFCSEnergyParametrization.cxx+").c_str());
    gROOT->LoadMacro((PATH + "ISF_FastCaloSimEvent/src/TFCSLateralShapeParametrization.cxx+").c_str());
    gROOT->LoadMacro((PATH + "ISF_FastCaloSimEvent/src/TFCSParametrizationBinnedChain.cxx+").c_str());
    gROOT->LoadMacro((PATH + "ISF_FastCaloSimEvent/src/TFCSParametrizationEbinChain.cxx+").c_str());
    gROOT->LoadMacro((PATH + "ISF_FastCaloSimEvent/src/TFCSParametrizationPDGIDSelectChain.cxx+").c_str());
    gROOT->LoadMacro((PATH + "ISF_FastCaloSimEvent/src/TFCSParametrizationFloatSelectChain.cxx+").c_str());
    gROOT->LoadMacro((PATH + "ISF_FastCaloSimEvent/src/TFCSParametrizationEkinSelectChain.cxx+").c_str());
    gROOT->LoadMacro((PATH + "ISF_FastCaloSimEvent/src/TFCSParametrizationEtaSelectChain.cxx+").c_str());
    gROOT->LoadMacro((PATH + "ISF_FastCaloSimEvent/src/TFCSParametrizationAbsEtaSelectChain.cxx+").c_str());

    gROOT->LoadMacro((PATH + "ISF_FastCaloSimEvent/src/TFCSFunction.cxx+").c_str());
    gROOT->LoadMacro((PATH + "ISF_FastCaloSimEvent/src/TFCS1DFunction.cxx+").c_str());
    gROOT->LoadMacro((PATH + "ISF_FastCaloSimEvent/src/TFCS1DFunctionHistogram.cxx+").c_str());
    gROOT->LoadMacro((PATH + "ISF_FastCaloSimEvent/src/TFCS1DFunctionInt16Histogram.cxx+").c_str());
    gROOT->LoadMacro((PATH + "ISF_FastCaloSimEvent/src/TFCS1DFunctionInt32Histogram.cxx+").c_str());
    gROOT->LoadMacro((PATH + "ISF_FastCaloSimEvent/src/TFCS1DFunctionRegression.cxx+").c_str());
    gROOT->LoadMacro((PATH + "ISF_FastCaloSimEvent/src/TFCS1DFunctionRegressionTF.cxx+").c_str()); //
    gROOT->LoadMacro((PATH + "ISF_FastCaloSimEvent/src/TFCS1DFunctionSpline.cxx+").c_str());
    gROOT->LoadMacro((PATH + "ISF_FastCaloSimEvent/src/TFCS1DFunctionTemplateHistogram.cxx+").c_str());


    gROOT->LoadMacro((PATH + "ISF_FastCaloSimEvent/src/TFCS2DFunction.cxx+").c_str());
    gROOT->LoadMacro((PATH + "ISF_FastCaloSimEvent/src/TFCS2DFunctionHistogram.cxx+").c_str());
    gROOT->LoadMacro((PATH + "ISF_FastCaloSimEvent/src/IntArray.cxx+g").c_str());
    gROOT->LoadMacro((PATH + "ISF_FastCaloSimEvent/src/DoubleArray.cxx+g").c_str());
    gROOT->LoadMacro((PATH + "ISF_FastCaloSimEvent/src/TFCSPCAEnergyParametrization.cxx+g").c_str());
    gROOT->LoadMacro((PATH + "ISF_FastCaloSimEvent/src/TFCSEnergyBinParametrization.cxx+").c_str());

    gROOT->LoadMacro((PATH + "ISF_FastCaloSimEvent/src/TFCSLateralShapeParametrizationHitBase.cxx+").c_str());
    gROOT->LoadMacro((PATH + "ISF_FastCaloSimEvent/src/TFCSLateralShapeParametrizationHitChain.cxx+").c_str());
    gROOT->LoadMacro((PATH + "ISF_FastCaloSimEvent/src/TFCSCenterPositionCalculation.cxx+").c_str());
    gROOT->LoadMacro((PATH + "ISF_FastCaloSimEvent/src/TFCSHistoLateralShapeParametrization.cxx+").c_str());
    gROOT->LoadMacro((PATH + "ISF_FastCaloSimEvent/src/TFCSHistoLateralShapeParametrizationFCal.cxx+").c_str());
    gROOT->LoadMacro((PATH + "ISF_FastCaloSimEvent/src/TFCSLateralShapeParametrizationHitNumberFromE.cxx+").c_str());
    gROOT->LoadMacro((PATH + "ISF_FastCaloSimEvent/src/TFCSHitCellMapping.cxx+").c_str());
    gROOT->LoadMacro((PATH + "ISF_FastCaloSimEvent/src/TFCSHitCellMappingFCal.cxx+").c_str());
    gROOT->LoadMacro((PATH + "ISF_FastCaloSimEvent/src/TFCSHitCellMappingWiggle.cxx+").c_str());
    gROOT->LoadMacro((PATH + "ISF_FastCaloSimEvent/src/TFCSHitCellMappingWiggleEMB.cxx+").c_str());

    gROOT->LoadMacro((PATH + "ISF_FastCaloSimParametrization/Root/TFCSSimpleLateralShapeParametrization.cxx+").c_str());
    gROOT->LoadMacro((PATH + "ISF_FastCaloSimParametrization/Root/TFCSNNLateralShapeParametrization.cxx+").c_str());
    gROOT->LoadMacro((PATH + "ISF_FastCaloSimParametrization/src/CaloGeometry.cxx+").c_str());
    gROOT->LoadMacro((PATH + "ISF_FastCaloSimParametrization/src/CaloGeometryLookup.cxx+").c_str());

    gROOT->LoadMacro((PATH + "ISF_FastCaloSimParametrization/tools/FCAL_ChannelMap.cxx+").c_str());
    
    gROOT->LoadMacro( (FastCaloSimCommonPath + "/src/TFCSSampleDiscovery.cxx+").c_str() );

    gROOT->LoadMacro( (FastCaloSimAnalyzerPath+"/Root/TFCSAnalyzerBase.cxx+").c_str() );
    gROOT->LoadMacro( (FastCaloSimAnalyzerPath+"/Root/TFCSAnalyzerHelpers.cxx+").c_str() );

    gROOT->LoadMacro((EnergyParameterizationPath + "/src/TFCS1DRegression.cxx+").c_str());//
    gROOT->LoadMacro((EnergyParameterizationPath + "/src/TFCS1DFunctionFactory.cxx+").c_str());//

    gROOT->LoadMacro((EnergyParameterizationPath + "/src/TreeReader.cxx+").c_str());
    gROOT->LoadMacro((EnergyParameterizationPath + "/src/TFCSMakeFirstPCA.cxx+").c_str());
    gROOT->LoadMacro((EnergyParameterizationPath + "/src/TFCSApplyFirstPCA.cxx+").c_str());

    gROOT->LoadMacro((EnergyParameterizationPath  + "/src/secondPCA.cxx+").c_str());
    gROOT->LoadMacro((EnergyParameterizationPath  + "/src/TFCSEnergyParametrizationPCABinCalculator.cxx+").c_str());
    gROOT->LoadMacro((EnergyParameterizationPath  + "/src/EnergyParametrizationValidation.cxx+").c_str());

    gROOT->LoadMacro( (FastCaloSimAnalyzerPath+"/Root/CaloGeometryFromFile.cxx+").c_str() );
    gROOT->LoadMacro( (FastCaloSimAnalyzerPath+"/Root/TFCSValidationEnergy.cxx+").c_str() );
    gROOT->LoadMacro( (FastCaloSimAnalyzerPath+"/Root/TFCSValidationEnergyAndCells.cxx+").c_str() );
    gROOT->LoadMacro( (FastCaloSimAnalyzerPath+"/Root/TFCSValidationHitSpy.cxx+").c_str() );
    gROOT->LoadMacro( (FastCaloSimAnalyzerPath+"/Root/TFCSValidationEnergyAndHits.cxx+").c_str() );
    gROOT->LoadMacro( (FastCaloSimAnalyzerPath+"/Root/TFCSShapeValidation.cxx+").c_str() );
    gROOT->LoadMacro( (FastCaloSimAnalyzerPath+"/Root/TFCSWriteCellsToTree.cxx+").c_str() );
    gROOT->LoadMacro( (FastCaloSimAnalyzerPath+"/Root/TFCSVertexZPositionStudies.cxx+").c_str() );


    gROOT->LoadMacro( (FastCaloSimAnalyzerPath+"/macro/FCS_dsid.cxx+").c_str() );


    if (1 == 0) {
        THtml html;
        html.SetProductName("TFCS");
        html.SetInputDir((PATH + "ISF_FastCaloSimParametrization/:" + PATH + "ISF_FastCaloSimParametrization/tools/:" + PATH + "ISF_FastCaloSimEvent/:"+FastCaloSimAnalyzerPath+"/FastCaloSimAnalyzer/").c_str());
        html.MakeAll();
    }
}
