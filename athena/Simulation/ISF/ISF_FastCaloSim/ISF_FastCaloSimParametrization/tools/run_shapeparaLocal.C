/*
  Copyright (C) 2002-2017 CERN for the benefit of the ATLAS collaboration
*/

/**********************************************************************************
 * Generates TTree and shower shape distributions for input of the NN regression
 * Single particle Geant4 events used as the input of this macro.
 * To run:
 * > .x init_shapepara.C+(1)
 * > .x run_shapeparaLocal.C("thin"/"bin"/"fit"/"plot")
 * <a.hasib@cern.ch>
 *********************************************************************************/
#include "TChain.h"
#include "TFile.h"
#include "TTree.h"

#include <iostream>
#include <stdlib.h>


using namespace std;

void run_shapeparaLocal(string action);


void run_shapeparaLocal(string action)
{
   // * set the input parameters -----------------------------------------------------------------------------

   string particle = "pion"; // pion, el_1mm, el_opt, photon
   float  energy   = 50;     // in GeV
   float  etamin   = 0.20;
   float  etamax   = 0.25;


   int   calolayer   = 2;
   int   PCAbin      = 1;
   int   nbinsAlpha  = 8;
   int   nbinsR      = 20;
   float mincalosize = -1000;     //whaaat are thooose?
   float tolerance   = .000000001; // for empty bin check



   // * regression parameters
   int neurons = 4;



   // * Create topDir and fileName strings

   std::string topDir   = "../../run/output/shape_para/";
   std::string fileName = particle + "_" + std::to_string(energy) + "GeV" + "_eta_" + std::to_string(etamin) + "_" + std::to_string(etamax) + "_layer" + std::to_string(calolayer) + "_PCAbin" + std::to_string(PCAbin);



   // * Create output directory

   system(("mkdir -p " + topDir + particle).c_str());

   //----------------------------------------------------------------------------------------------------------------

   // * set what you want to run

   bool doThinning, doStudy, doBinning, doRegression, doPlotting;

   if (action.compare("thin") == 0)
   {
      doThinning   = true;
      doStudy      = false;
      doBinning    = false;
      doRegression = false;
      doPlotting   = false;
   }
   else if (action.compare("study") == 0)
   {
      doThinning   = false;
      doStudy      = true;
      doBinning    = false;
      doRegression = false;
      doPlotting   = false;
   }
   else if (action.compare("bin") == 0)
   {
      doThinning   = false;
      doStudy      = false;
      doBinning    = true;
      doRegression = false;
      doPlotting   = false;
   }
   else if (action.compare("fit") == 0)
   {
      doThinning   = false;
      doStudy      = false;
      doBinning    = false;
      doRegression = true;
      doPlotting   = false;
   }
   else if (action.compare("plot") == 0)
   {
      doThinning   = false;
      doStudy      = false;
      doBinning    = false;
      doRegression = false;
      doPlotting   = true;
   }



   // * Run ShowerShapeBinning...
   if (doThinning)
   {
      // * configure particle type and input samples

      cout << " Running on local machine. Cannot run ShowerShapeThinning  ...." << endl;
      exit(EXIT_FAILURE);
   }
   else if (doStudy)
   {
      // std::string fileallPCA = particle + "_" + std::to_string(energy) + "GeV" + "_eta_" + std::to_string(etamin) + "_" + std::to_string(etamax) + "_layer" + std::to_string(calolayer);

      std::string HitsFile = topDir + particle + "/Hits_" + fileName + ".root";

      ShowerShapeStudy *shapeStudy = new ShowerShapeStudy();

      shapeStudy->set_calolayer(calolayer);
      shapeStudy->set_PCAbin(PCAbin);
      shapeStudy->set_nbinsR(nbinsR);
      shapeStudy->set_nbinsAlpha(nbinsAlpha);
      shapeStudy->set_mincalosize(mincalosize);
      shapeStudy->set_particle(particle);
      shapeStudy->set_energy(energy);
      shapeStudy->set_eta(etamin, etamax);
      shapeStudy->set_tolerance(tolerance);
      shapeStudy->set_topDir(topDir);
      shapeStudy->set_fileName(fileName);
      shapeStudy->set_hitsNtupleName(HitsFile);

      //shapeStudy->InvesitageShowerCenter(HitsFile);

      shapeStudy->EachParticleShower();
   }
   else if (doBinning)
   {
      std::string HitsAlphaDrFile = topDir + particle + "/HitsAlphaDr_" + fileName + ".root";
      std::ifstream hitsalphadrfile(HitsAlphaDrFile);

      // std::string fileallPCA = particle + "_" + std::to_string(energy) + "GeV" + "_eta_" + std::to_string(etamin) + "_" + std::to_string(etamax) + "_layer" + std::to_string(calolayer);
      std::string HitsFile = topDir + particle + "/Hits_" + fileName + ".root";


      ShowerShapeBinning *shapeBin = new ShowerShapeBinning();

      shapeBin->set_calolayer(calolayer);
      shapeBin->set_PCAbin(PCAbin);
      shapeBin->set_nbinsR(nbinsR);
      shapeBin->set_nbinsAlpha(nbinsAlpha);
      shapeBin->set_mincalosize(mincalosize);
      shapeBin->set_particle(particle);
      shapeBin->set_energy(energy);
      shapeBin->set_eta(etamin, etamax);
      shapeBin->set_tolerance(tolerance);
      shapeBin->set_topDir(topDir);
      shapeBin->set_fileName(fileName);

      bool force = true;

      if (!hitsalphadrfile)
      {
         cout << " HitsAlphaDr root file is missing ....Creating..." << endl;
         shapeBin->CreateHitsAlphaDrNtuple(HitsFile);
         cout << " Created HitsAlphaDrNtuple at " << HitsAlphaDrFile << endl;
      }
      else
      {
         cout << " * HitsAlphaDrNtuple already exits!! Using it to RunBinning()..." << endl;
         shapeBin->RunBinning();
         cout << " * Creating TTree to be used as input of NN....." << endl;
         shapeBin->CreateNNinput();
         cout << " * NNinput file is saved in the directory = " << topDir.c_str() << endl;
      }
   }
   else if (doRegression)
   {
      // * NN regression ....

      cout << " * Running NN regression ..." << endl;

      std::string NNinputName  = topDir + particle + "/NNinput_"/*nbinsR" + std::to_string(nbinsR) + "_"*/ + fileName + ".root";
      std::string NNoutputName = topDir + particle + "/NNoutput_neurons" + std::to_string(neurons) + "_nbinsR" + std::to_string(nbinsR) + fileName + ".root";

      ShowerShapeRegression *shapeRegression = new ShowerShapeRegression();

      shapeRegression->set_calolayer(calolayer);
      shapeRegression->set_PCAbin(PCAbin);
      shapeRegression->set_nbinsR(nbinsR);
      shapeRegression->set_particle(particle);
      shapeRegression->set_energy(energy);
      shapeRegression->set_eta(etamin, etamax);
      shapeRegression->set_topDir(topDir);
      shapeRegression->set_fileName(fileName);
      shapeRegression->set_NNinputName(NNinputName);
      shapeRegression->set_NNoutputName(NNoutputName);
      shapeRegression->set_neurons(neurons);

      std::vector < string > targetVarVec;

      //targetVarVec.push_back("Hits");
      // targetVarVec.push_back("Energy");
      //targetVarVec.push_back("EnergyDensity");
      //targetVarVec.push_back("LnEnergy");
      //targetVarVec.push_back("LnEnergyDensity");
      //targetVarVec.push_back("EnergyNorm");
      targetVarVec.push_back("hEnergyNorm");
      shapeRegression->Run(targetVarVec);
   }
   else if (doPlotting)
   {
      // * plotting stuff...
      cout << " * Running plotting macro ...." << endl;

      // * create a plots directory
      system(("mkdir -p " + topDir + particle + "/plots"/*_nbinsR" + std::to_string(nbinsR)*/ + "_layer" + std::to_string(calolayer) + "_pca" + std::to_string(PCAbin) + "/").c_str());


      system(("mkdir -p " + topDir + particle + "/plot_books/").c_str());

      std::string outputDirName = topDir + particle + "/plots"/*_nbinsR" + std::to_string(nbinsR)*/ + "_layer" + std::to_string(calolayer) + "_pca" + std::to_string(PCAbin) + "/";

      std::string NNinputName  = topDir + particle + "/NNinput_"/*nbinsR" + std::to_string(nbinsR) + "_"*/ + fileName + ".root";
      std::string NNoutputName = topDir + particle + "/NNoutput_neurons" + std::to_string(neurons) + "_nbinsR" + std::to_string(nbinsR) + fileName + ".root";


      ShowerShapePlotting *shapePlot = new ShowerShapePlotting();

      shapePlot->set_calolayer(calolayer);
      shapePlot->set_PCAbin(PCAbin);
      shapePlot->set_nbinsR(nbinsR);
      shapePlot->set_particle(particle);
      shapePlot->set_energy(energy);
      shapePlot->set_eta(etamin, etamax);
      shapePlot->set_topDir(topDir);
      shapePlot->set_fileName(fileName);
      shapePlot->set_outputDirName(outputDirName);
      shapePlot->set_NNinputName(NNinputName);
      shapePlot->set_NNoutputName(NNoutputName);
      shapePlot->set_neurons(neurons);


      // * plotting methods

      std::vector < string > histVec;
      //histVec.push_back("hHits");
      //histVec.push_back("hEnergy");
      // histVec.push_back("hEnergyDensity");
      //histVec.push_back("hLnEnergy");
      //histVec.push_back("hLnEnergyDensity");
      histVec.push_back("hEnergyNorm");

      shapePlot->PlotEnergyDensityGradient();
      shapePlot->PlotPolar(histVec, false);
      //shapePlot->CreateValidationPlot();
      // shapePlot->CreateHTML(histVec);
      shapePlot->CreatePlotBook(histVec);
   }
   else
   {
      cout << " Choose .x run_shapepara.C(0) to run ShowerShapeBinning or .x run_shapepara(1) to run ShowerShapePlotting" << endl;
      exit(EXIT_FAILURE);
   }
}
