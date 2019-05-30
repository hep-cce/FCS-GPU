#include "TROOT.h"

void run() {
  
  std::string PATH = "../../athena/Simulation/ISF/ISF_FastCaloSim/";
  std::string arguments=Form("\"%s\"",PATH.c_str());
  
  gROOT->ProcessLine((".x initTFCSAnalyzer.C("+arguments+")").c_str());
  
  int dsid=100035;
  int dsid_zv0=100035;
  //std::string sampleData = "/eos/atlas/atlascerngroupdisk/proj-simul/InputSamplesSummer18_complete_list.txt";
  std::string sampleData = "mySampleList.txt";
  std::string topDir = "./output/";
  std::string version = "ver01";
  float energy_cutoff = 0.9995;
  std::string topPlotDir = "output_plot/";
  bool do2Dparameterization = true;
  bool isPhisymmetry = true;
  bool doZVertexStudies=true;
  bool doMeanRz = false;
  bool useMeanRz = true;
  //bool doMeanRz = true;
  //bool useMeanRz = false;
  
  arguments=Form("%i,%i,\"%s\",\"%s\",\"%s\",%f,\"%s\",%i,%i,%i,%i,%i",dsid,dsid_zv0,sampleData.c_str(),topDir.c_str(),version.c_str(),energy_cutoff,topPlotDir.c_str(),do2Dparameterization,isPhisymmetry,doMeanRz,useMeanRz,doZVertexStudies);
  
  std::cout << "Running with parameters: " << arguments << endl;
  
  //runTFCS2DParametrizationHistogram(int dsid = 431004, int dsid_zv0 = -999,  std::string sampleData = "../python/inputSampleList.txt", std::string topDir = "./output/"
  //, std::string version = "ver01", float energy_cutoff = 0.9995,  std::string topPlotDir = "output_plot/", bool do2DParam = true, bool isPhisymmetry = true,  bool doMaxRz = false, bool doZVertexStudies = false)
  
  gROOT->ProcessLine((".x runTFCS2DParametrizationHistogram.cxx(" + arguments +  ")").c_str()); 
}
