/*
  Copyright (C) 2002-2021 CERN for the benefit of the ATLAS collaboration
*/

#include "TROOT.h"

void runZVertexStudies(int dsid,int dsid_zv0,bool do2Dparameterization,bool isPhisymmetry,bool doMeanRz,bool useMeanRz,bool doZVertexStudies) {
  
  std::string PATH = "../../athena/Simulation/ISF/ISF_FastCaloSim/";
  std::string arguments=Form("\"%s\"",PATH.c_str());
  
  gROOT->ProcessLine((".x initTFCSAnalyzer.C("+arguments+")").c_str());
  
  //std::string sampleData = "inputSampleList.txt";
  std::string sampleData = "mySampleList.txt";
  std::string topDir = "./output/";
  std::string version = "ver01";
  float energy_cutoff = 0.9995;
  std::string topPlotDir = "output_plot/";
  
  arguments=Form("%i,%i,\"%s\",\"%s\",\"%s\",%f,\"%s\",%i,%i,%i,%i,%i",dsid,dsid_zv0,sampleData.c_str(),topDir.c_str(),version.c_str(),energy_cutoff,topPlotDir.c_str(),do2Dparameterization,isPhisymmetry,doMeanRz,useMeanRz,doZVertexStudies);
  
  std::cout << "Running with parameters: " << arguments << endl;
  
  //runTFCS2DParametrizationHistogram(int dsid = 431004, int dsid_zv0 = -999,  std::string sampleData = "../python/inputSampleList.txt", std::string topDir = "./output/"
  //, std::string version = "ver01", float energy_cutoff = 0.9995,  std::string topPlotDir = "output_plot/", bool do2DParam = true, bool isPhisymmetry = true,  bool doMaxRz = false, bool doZVertexStudies = false)
  
  gROOT->ProcessLine((".x runTFCS2DParametrizationHistogram.cxx(" + arguments +  ")").c_str()); 
}
