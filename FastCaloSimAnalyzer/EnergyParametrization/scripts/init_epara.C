/*
  Copyright (C) 2002-2023 CERN for the benefit of the ATLAS collaboration
*/

#include "TSystem.h"
#include "TROOT.h"
#include "TEnv.h"
#include "TLorentzVector.h"
#include "TMatrixD.h"
#include <iostream>

void init_epara();

void init_epara()
{
 
 string ATHENA_PATH="../../../../../../athena/";
 
 cout<<endl;
 cout<<"***** Init energy parametrisation & validation *****"<<endl;
 cout<<endl;
 cout<<"Path to the Athena packages: "<<ATHENA_PATH<<endl;
 cout<<endl;
 
 TLorentzVector *t;
 TMatrixD *m;
 
 gEnv->SetValue("ACLiC.IncludePaths",TString(gEnv->GetValue("ACLiC.IncludePaths",""))+" -D__FastCaloSimStandAlone__ ");
 
 gInterpreter->AddIncludePath("..");
 gInterpreter->AddIncludePath("../src");
 gInterpreter->AddIncludePath("../../FastCaloSimAnalyzer");
 gInterpreter->AddIncludePath("../../FastCaloSimAnalyzer/CLHEP");
 gInterpreter->AddIncludePath("../../FastCaloSimAnalyzer/CLHEP/Random");
 gInterpreter->AddIncludePath("../../FastCaloSimAnalyzer/FastCaloSimAnalyzer");

 gROOT->LoadMacro("../../FastCaloSimAnalyzer/CLHEP/Random/RandomEngine.cxx+");
 gROOT->LoadMacro("../../FastCaloSimAnalyzer/CLHEP/Random/TRandomEngine.cxx+");
 gROOT->LoadMacro("../../FastCaloSimAnalyzer/CLHEP/Random/RandFlat.cxx+");
 gROOT->LoadMacro("../../FastCaloSimAnalyzer/CLHEP/Random/RandGauss.cxx+");
 gROOT->LoadMacro("../../FastCaloSimAnalyzer/CLHEP/Random/RandPoisson.cxx+");

 gInterpreter->AddIncludePath((ATHENA_PATH+"Simulation/ISF/ISF_FastCaloSim/ISF_FastCaloSimEvent").c_str());
 gInterpreter->AddIncludePath((ATHENA_PATH+"Simulation/ISF/ISF_FastCaloSim/ISF_FastCaloSimEvent/src").c_str());
 gInterpreter->AddIncludePath((ATHENA_PATH+"Calorimeter/CaloGeoHelpers/").c_str());
 
 //be careful, the order of these calls seems to matter. If changed, the strangest and non-explainable things happen :(
 //gROOT->LoadMacro("../src/TFCSEpara.cxx+");
 
 
 gROOT->LoadMacro("../src/TreeReader.cxx+");
 gROOT->LoadMacro((ATHENA_PATH+"Simulation/ISF/ISF_FastCaloSim/ISF_FastCaloSimEvent/src/TFCSSimulationState.cxx+").c_str());

 gROOT->LoadMacro("../src/TFCSMakeFirstPCA.cxx+");
 gROOT->LoadMacro("../src/TFCSApplyFirstPCA.cxx+");
 gROOT->LoadMacro((ATHENA_PATH+"Simulation/ISF/ISF_FastCaloSim/ISF_FastCaloSimEvent/src/IntArray.cxx+").c_str());
 gROOT->LoadMacro((ATHENA_PATH+"Simulation/ISF/ISF_FastCaloSim/ISF_FastCaloSimEvent/src/TFCS1DFunction.cxx+").c_str());
 gROOT->LoadMacro((ATHENA_PATH+"Simulation/ISF/ISF_FastCaloSim/ISF_FastCaloSimEvent/src/TFCSFunction.cxx+").c_str());
 gROOT->LoadMacro((ATHENA_PATH+"Simulation/ISF/ISF_FastCaloSim/ISF_FastCaloSimEvent/src/TFCS1DFunctionRegression.cxx+").c_str());
 gROOT->LoadMacro((ATHENA_PATH+"Simulation/ISF/ISF_FastCaloSim/ISF_FastCaloSimEvent/src/TFCS1DFunctionRegressionTF.cxx+").c_str());
 gROOT->LoadMacro((ATHENA_PATH+"Simulation/ISF/ISF_FastCaloSim/ISF_FastCaloSimEvent/src/TFCS1DFunctionHistogram.cxx+").c_str());
 gROOT->LoadMacro((ATHENA_PATH+"Simulation/ISF/ISF_FastCaloSim/ISF_FastCaloSimEvent/src/TFCSExtrapolationState.cxx+").c_str());
 gROOT->LoadMacro((ATHENA_PATH+"Simulation/ISF/ISF_FastCaloSim/ISF_FastCaloSimEvent/src/TFCSTruthState.cxx+").c_str());
 gROOT->LoadMacro((ATHENA_PATH+"Simulation/ISF/ISF_FastCaloSim/ISF_FastCaloSimEvent/src/TFCSParametrizationBase.cxx+").c_str());
 gROOT->LoadMacro((ATHENA_PATH+"Simulation/ISF/ISF_FastCaloSim/ISF_FastCaloSimEvent/src/TFCSParametrization.cxx+").c_str());
 gROOT->LoadMacro((ATHENA_PATH+"Simulation/ISF/ISF_FastCaloSim/ISF_FastCaloSimEvent/src/TFCSEnergyParametrization.cxx+").c_str());
 gROOT->LoadMacro((ATHENA_PATH+"Simulation/ISF/ISF_FastCaloSim/ISF_FastCaloSimEvent/src/TFCSPCAEnergyParametrization.cxx+").c_str());
 gROOT->LoadMacro("../src/TFCSEnergyParametrizationPCABinCalculator.cxx+");
 gROOT->LoadMacro((ATHENA_PATH+"Simulation/ISF/ISF_FastCaloSim/ISF_FastCaloSimEvent/src/TFCS1DFunction.cxx+").c_str());
 gROOT->LoadMacro((ATHENA_PATH+"Simulation/ISF/ISF_FastCaloSim/ISF_FastCaloSimEvent/src/TFCS1DFunctionRegression.cxx+").c_str());
 gROOT->LoadMacro((ATHENA_PATH+"Simulation/ISF/ISF_FastCaloSim/ISF_FastCaloSimEvent/src/TFCS1DFunctionRegressionTF.cxx+").c_str());
 gROOT->LoadMacro((ATHENA_PATH+"Simulation/ISF/ISF_FastCaloSim/ISF_FastCaloSimEvent/src/TFCS1DFunctionHistogram.cxx+").c_str());
 gROOT->LoadMacro("../src/TFCS1DRegression.cxx+");
 gROOT->LoadMacro("../src/TFCS1DFunctionFactory.cxx+"); 
 gROOT->LoadMacro("../src/secondPCA.cxx+");
 gROOT->LoadMacro("../src/EnergyParametrizationValidation.cxx+");
 
 
 gROOT->LoadMacro("../../FastCaloSimAnalyzer/Root/TFCSAnalyzerBase.cxx+");
 
 
 //init the DSID db:
 gROOT->LoadMacro("../src/DSIDConverter.cxx+");
 
 cout<<"init done"<<endl;
 
}

