/*
  Copyright (C) 2002-2017 CERN for the benefit of the ATLAS collaboration
*/

/**********************************************************************************
* Generates TTree and shower shape distributions for input of the NN regression
* Single particle Geant4 events used as the input of this macro.
* To run:
* > .x init_shapepara.C+(0/1) 0 for lxplus 1 for local
* > .x run_shapepara.C("thin"/"bin"/"fit"/"plot")
* <a.hasib@cern.ch>
*********************************************************************************/
#include "TChain.h"
#include "TFile.h"
#include "TTree.h"

#include <iostream>
#include <stdlib.h>


using namespace std;

void run_shapepara(int layer, int binpca);


void run_shapepara(int layer, int binpca)
{
// * set the input parameters -----------------------------------------------------------------------------

   string particle = "pion"; // pion, el_1mm, el_opt, photon
   float  energy   = 50;     // in GeV
   float  etamin   = 0.20;
   float  etamax   = 0.25;


   int   calolayer   = layer;
   int   PCAbin      = binpca;
   int   nbinsR      = 20;
   int   nbinsAlpha  = 8;
   float mincalosize = -1000;     //whaaat are thooose?
   float tolerance   = .00001;    // for empty bin check



   // * regression parameters
   int neurons = 12;


   // * Create topDir and fileName strings

   std::string topDir   = "../../run/output/shape_para/";
   std::string fileName = particle + "_" + std::to_string(energy) + "GeV" + "_eta_" + std::to_string(etamin) + "_" + std::to_string(etamax) + "_layer" + std::to_string(calolayer) + "_PCAbin" + std::to_string(PCAbin);



   // * Create output directory

   system(("mkdir -p " + topDir + particle).c_str());

   //----------------------------------------------------------------------------------------------------------------

   // * set what you want to run
   //


   bool doThinning, doBinning, doRegression, doPlotting;

   // if (action.compare("thin") == 0)
   // {
   //    doThinning   = true;
   //    doBinning    = false;
   //    doRegression = false;
   //    doPlotting   = false;
   // }
   // else if (action.compare("bin") == 0)
   // {
   //    doThinning   = false;
   //    doBinning    = true;
   //    doRegression = false;
   //    doPlotting   = false;
   // }
   // else if (action.compare("fit") == 0)
   // {
   //    doThinning   = false;
   //    doBinning    = false;
   //    doRegression = true;
   //    doPlotting   = false;
   // }
   // else if (action.compare("plot") == 0)
   // {
   //    doThinning   = false;
   //    doBinning    = false;
   //    doRegression = false;
   //    doPlotting   = true;
   // }

   doThinning   = true;
   doBinning    = false;
   doRegression = false;
   doPlotting   = false;

   // * Run ShowerShapeBinning...
   if (doThinning)
   {
      // * configure particle type and input samples

      cout << " ** configure particle type and input samples ...." << endl;

      vector < string > input;      // vector of input samples.
      vector < string > pca;        // vector of input pca files



      if (particle == "pion")
      {
         // * input sample
         // input.push_back("/afs/cern.ch/work/a/ahasib/FastCaloSim/old/StandAlone/ISF_FastCaloSimParametrization_WorkingBranch/ISF_HitAnalysis6_evgen_calo__211_E50000_50000_eta20_25_Evts0-5500_vz_0_origin_calo.merged.pool.root");
         input.push_back("/afs/cern.ch/user/a/ahasib/public/NewInputSamples/pion_50GeV_eta_020_025.root");

         // * PCA file
         pca.push_back("/afs/cern.ch/user/a/ahasib/WorkDir/FastCaloSim/ISF_FastCaloSim/PCAs/pion/firstPCA.root");
      }
      else if (particle == "el_1mm")
      {
         // * input sample

         input.push_back("root://eosatlas//eos/atlas/user/s/schaarsc/FCS/user.fladias.428137.FastCalo_pid11_E65536_etam35_35_zv_m100.e4001_s2865_r7736.w0_162706_matched_output.root/user.fladias.8834800._000001.matched_output.root");
         input.push_back("root://eosatlas//eos/atlas/user/s/schaarsc/FCS/user.fladias.428137.FastCalo_pid11_E65536_etam35_35_zv_m100.e4001_s2865_r7736.w0_162706_matched_output.root/user.fladias.8834800._000002.matched_output.root");
         input.push_back("root://eosatlas//eos/atlas/user/s/schaarsc/FCS/user.fladias.428137.FastCalo_pid11_E65536_etam35_35_zv_m100.e4001_s2865_r7736.w0_162706_matched_output.root/user.fladias.8834800._000003.matched_output.root");
         input.push_back("root://eosatlas//eos/atlas/user/s/schaarsc/FCS/user.fladias.428137.FastCalo_pid11_E65536_etam35_35_zv_m100.e4001_s2865_r7736.w0_162706_matched_output.root/user.fladias.8834800._000004.matched_output.root");
         input.push_back("root://eosatlas//eos/atlas/user/s/schaarsc/FCS/user.fladias.428137.FastCalo_pid11_E65536_etam35_35_zv_m100.e4001_s2865_r7736.w0_162706_matched_output.root/user.fladias.8834800._000005.matched_output.root");
         input.push_back("root://eosatlas//eos/atlas/user/s/schaarsc/FCS/user.fladias.428137.FastCalo_pid11_E65536_etam35_35_zv_m100.e4001_s2865_r7736.w0_162706_matched_output.root/user.fladias.8834800._000006.matched_output.root");
         input.push_back("root://eosatlas//eos/atlas/user/s/schaarsc/FCS/user.fladias.428137.FastCalo_pid11_E65536_etam35_35_zv_m100.e4001_s2865_r7736.w0_162706_matched_output.root/user.fladias.8834800._000007.matched_output.root");
         input.push_back("root://eosatlas//eos/atlas/user/s/schaarsc/FCS/user.fladias.428137.FastCalo_pid11_E65536_etam35_35_zv_m100.e4001_s2865_r7736.w0_162706_matched_output.root/user.fladias.8834800._000008.matched_output.root");
         input.push_back("root://eosatlas//eos/atlas/user/s/schaarsc/FCS/user.fladias.428137.FastCalo_pid11_E65536_etam35_35_zv_m100.e4001_s2865_r7736.w0_162706_matched_output.root/user.fladias.8834800._000009.matched_output.root");
         input.push_back("root://eosatlas//eos/atlas/user/s/schaarsc/FCS/user.fladias.428137.FastCalo_pid11_E65536_etam35_35_zv_m100.e4001_s2865_r7736.w0_162706_matched_output.root/user.fladias.8834800._000010.matched_output.root");
         input.push_back("root://eosatlas//eos/atlas/user/s/schaarsc/FCS/user.fladias.428137.FastCalo_pid11_E65536_etam35_35_zv_m100.e4001_s2865_r7736.w0_162706_matched_output.root/user.fladias.8834800._000011.matched_output.root");
         input.push_back("root://eosatlas//eos/atlas/user/s/schaarsc/FCS/user.fladias.428137.FastCalo_pid11_E65536_etam35_35_zv_m100.e4001_s2865_r7736.w0_162706_matched_output.root/user.fladias.8834800._000012.matched_output.root");
         input.push_back("root://eosatlas//eos/atlas/user/s/schaarsc/FCS/user.fladias.428137.FastCalo_pid11_E65536_etam35_35_zv_m100.e4001_s2865_r7736.w0_162706_matched_output.root/user.fladias.8834800._000013.matched_output.root");
         input.push_back("root://eosatlas//eos/atlas/user/s/schaarsc/FCS/user.fladias.428137.FastCalo_pid11_E65536_etam35_35_zv_m100.e4001_s2865_r7736.w0_162706_matched_output.root/user.fladias.8834800._000014.matched_output.root");
         input.push_back("root://eosatlas//eos/atlas/user/s/schaarsc/FCS/user.fladias.428137.FastCalo_pid11_E65536_etam35_35_zv_m100.e4001_s2865_r7736.w0_162706_matched_output.root/user.fladias.8834800._000015.matched_output.root");
         input.push_back("root://eosatlas//eos/atlas/user/s/schaarsc/FCS/user.fladias.428137.FastCalo_pid11_E65536_etam35_35_zv_m100.e4001_s2865_r7736.w0_162706_matched_output.root/user.fladias.8834800._000016.matched_output.root");
         input.push_back("root://eosatlas//eos/atlas/user/s/schaarsc/FCS/user.fladias.428137.FastCalo_pid11_E65536_etam35_35_zv_m100.e4001_s2865_r7736.w0_162706_matched_output.root/user.fladias.8834800._000017.matched_output.root");
         input.push_back("root://eosatlas//eos/atlas/user/s/schaarsc/FCS/user.fladias.428137.FastCalo_pid11_E65536_etam35_35_zv_m100.e4001_s2865_r7736.w0_162706_matched_output.root/user.fladias.8834800._000018.matched_output.root");
         input.push_back("root://eosatlas//eos/atlas/user/s/schaarsc/FCS/user.fladias.428137.FastCalo_pid11_E65536_etam35_35_zv_m100.e4001_s2865_r7736.w0_162706_matched_output.root/user.fladias.8834800._000019.matched_output.root");
         input.push_back("root://eosatlas//eos/atlas/user/s/schaarsc/FCS/user.fladias.428137.FastCalo_pid11_E65536_etam35_35_zv_m100.e4001_s2865_r7736.w0_162706_matched_output.root/user.fladias.8834800._000020.matched_output.root");
         input.push_back("root://eosatlas//eos/atlas/user/s/schaarsc/FCS/user.fladias.428137.FastCalo_pid11_E65536_etam35_35_zv_m100.e4001_s2865_r7736.w0_162706_matched_output.root/user.fladias.8834800._000021.matched_output.root");

         // * PCA file

         pca.push_back("/afs/cern.ch/user/a/ahasib/WorkDir/FastCaloSim/ISF_FastCaloSim/PCAs/electron/firstPCA_el_s2865.root");
      }
      else if (particle == "el_opt")
      {
         // * optimized merge samples

         input.push_back("root://eosatlas//eos/atlas/user/s/schaarsc/FCS/user.fladias.428137.FastCalo_pid11_E65536_etam35_35_zv_m100.e4001_s2864_r7736.w0_162706_matched_output.root/user.fladias.8834798._000001.matched_output.root");
         input.push_back("root://eosatlas//eos/atlas/user/s/schaarsc/FCS/user.fladias.428137.FastCalo_pid11_E65536_etam35_35_zv_m100.e4001_s2864_r7736.w0_162706_matched_output.root/user.fladias.8834798._000002.matched_output.root");
         input.push_back("root://eosatlas//eos/atlas/user/s/schaarsc/FCS/user.fladias.428137.FastCalo_pid11_E65536_etam35_35_zv_m100.e4001_s2864_r7736.w0_162706_matched_output.root/user.fladias.8834798._000003.matched_output.root");
         input.push_back("root://eosatlas//eos/atlas/user/s/schaarsc/FCS/user.fladias.428137.FastCalo_pid11_E65536_etam35_35_zv_m100.e4001_s2864_r7736.w0_162706_matched_output.root/user.fladias.8834798._000004.matched_output.root");
         input.push_back("root://eosatlas//eos/atlas/user/s/schaarsc/FCS/user.fladias.428137.FastCalo_pid11_E65536_etam35_35_zv_m100.e4001_s2864_r7736.w0_162706_matched_output.root/user.fladias.8834798._000005.matched_output.root");
         input.push_back("root://eosatlas//eos/atlas/user/s/schaarsc/FCS/user.fladias.428137.FastCalo_pid11_E65536_etam35_35_zv_m100.e4001_s2864_r7736.w0_162706_matched_output.root/user.fladias.8834798._000006.matched_output.root");
         input.push_back("root://eosatlas//eos/atlas/user/s/schaarsc/FCS/user.fladias.428137.FastCalo_pid11_E65536_etam35_35_zv_m100.e4001_s2864_r7736.w0_162706_matched_output.root/user.fladias.8834798._000007.matched_output.root");
         input.push_back("root://eosatlas//eos/atlas/user/s/schaarsc/FCS/user.fladias.428137.FastCalo_pid11_E65536_etam35_35_zv_m100.e4001_s2864_r7736.w0_162706_matched_output.root/user.fladias.8834798._000008.matched_output.root");


         // * PCA file

         pca.push_back("/afs/cern.ch/user/a/ahasib/WorkDir/FastCaloSim/ISF_FastCaloSim/PCAs/electron/firstPCA_el_s2864.root");
      }
      else if (particle == "photon")
      {
         // * input sample
         input.push_back("/afs/cern.ch/work/a/ahasib/FastCaloSim/old/StandAlone/ISF_FastCaloSimParametrization_WorkingBranch/photon/ISF_HitAnalysis_evgen_calo__22_E50000_50000_eta20_25_Evts0-5500_vz_0_origin_calo.merged.pool.root");

         // * PCA file
         pca.push_back("/afs/cern.ch/user/a/ahasib/WorkDir/FastCaloSim/ISF_FastCaloSim/PCAs/photon/firstPCA.root");
      }
      else
      {
         cout << "Error:: Particle type not configured! Exiting..." << endl;
         exit(EXIT_FAILURE);
      }


      // * add the input files in TChain

      TChain *mychain = new TChain("FCS_ParametrizationInput");

      for (auto i : input)
      {
         mychain->Add(i.c_str());
      }

      cout << " * Prepare to run on: " << particle << " with entries = " << mychain->GetEntries() << endl;

      TFile *PCAfile = TFile::Open((pca.at(0)).c_str());           //for the 1st pca file

      if (!PCAfile)
      {
         cout << "Error:: Cannot locate the PCA file..." << endl;
         exit(EXIT_FAILURE);
      }

      TTree *TPCA = (TTree *)PCAfile->Get("tree_1stPCA");

      if (!TPCA)
      {
         cout << "Error:: Cannot locate the PCA TTree..." << endl;
         exit(EXIT_FAILURE);
      }

      cout << " * Using PCA file = " << (pca.at(0)).c_str() << " with entries = " << TPCA->GetEntries() << endl;

      // * Set the required parameter values

      ShowerShapeThinning *shapeThin = new ShowerShapeThinning(mychain, TPCA);

      shapeThin->set_calolayer(calolayer);
      shapeThin->set_PCAbin(PCAbin);
      shapeThin->set_nbinsR(nbinsR);
      shapeThin->set_nbinsAlpha(nbinsAlpha);
      shapeThin->set_mincalosize(mincalosize);
      shapeThin->set_particle(particle);
      shapeThin->set_energy(energy);
      shapeThin->set_eta(etamin, etamax);
      shapeThin->set_tolerance(tolerance);
      shapeThin->set_topDir(topDir);
      shapeThin->set_fileName(fileName);



      // * file to save hits
      // std::string fileallPCA = particle + "_" + std::to_string(energy) + "GeV" + "_eta_" + std::to_string(etamin) + "_" + std::to_string(etamax) + "_layer" + std::to_string(calolayer);

      std::string HitsFile = topDir + particle + "/Hits_" + fileName + ".root";
      shapeThin->set_hitsNtupleName(HitsFile);

      std::string HitsAlphaDrFile = topDir + particle + "/HitsAlphaDr_" + fileName + ".root";


      // * check to see it the hits ntuple exits
      std::ifstream hitsfile(HitsFile);
      std::ifstream hitsalphadrfile(HitsAlphaDrFile);

      bool force = true;

      if (!hitsfile or force)
      {
         cout << " * HitsNtuple doesn't exits. Creating from input samples...." << endl;
         TFile *hitsFile = new TFile(HitsFile.c_str(), "recreate");

         shapeThin->CreateHitsNtuple(hitsFile);
      }
      else if (hitsfile)
      {
         cout << " * HitsNtuple already exits!! Using it to CreateHitsAlphaDrNtuple()...." << endl;
         shapeThin->CreateHitsAlphaDrNtuple(HitsFile);
         //shapeThin->InvesitageShowerCenter(HitsFile);
         // shapeThin->FindShowerCenter();
         cout << " * Finished creating HitsAlphaDrNtuple..." << endl;
      }
      else
      {
         cout << " Choose a method to run ...." << endl;
         exit(EXIT_FAILURE);
      }
   }
   else if (doBinning)
   {
      std::string HitsAlphaDrFile = topDir + particle + "/HitsAlphaDr_" + fileName + ".root";
      std::ifstream hitsalphadrfile(HitsAlphaDrFile);

      if (!hitsalphadrfile)
      {
         cout << " HitsAlphaDr root file is missing ...." << endl;
         exit(EXIT_FAILURE);
      }

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


      cout << " * HitsAlphaDrNtuple already exits!! Using it to RunBinning()..." << endl;
      shapeBin->RunBinning();
      cout << " * Creating TTree to be used as input of NN....." << endl;
      shapeBin->CreateNNinput();
      cout << " * NNinput file is saved in the directory = " << topDir.c_str() << endl;
   }
   else if (doRegression)
   {
      // * NN regression ....

      cout << " * Running NN regression ..." << endl;

      std::string NNinputName  = topDir + particle + "/NNinput_nbinsR" + std::to_string(nbinsR) + "_" + fileName + ".root";
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
      //targetVarVec.push_back("Energy");
      //targetVarVec.push_back("EnergyDensity");
      //targetVarVec.push_back("LnEnergy");
      targetVarVec.push_back("LnEnergyDensity");
      shapeRegression->Run(targetVarVec);
   }
   else if (doPlotting)
   {
      // * plotting stuff...
      cout << " * Running plotting macro ...." << endl;

      // * create a plots directory

      system(("mkdir -p " + topDir + particle + "/plots_nbinsR" + std::to_string(nbinsR) + "_neuron" + std::to_string(neurons) + "/").c_str());
      std::string outputDirName = topDir + particle + "/plots_nbinsR" + std::to_string(nbinsR) + "_neuron" + std::to_string(neurons) + "/";
      std::string NNinputName   = topDir + particle + "/NNinput_nbinsR" + std::to_string(nbinsR) + "_" + fileName + ".root";
      std::string NNoutputName  = topDir + particle + "/NNoutput_neurons" + std::to_string(neurons) + "_nbinsR" + std::to_string(nbinsR) + fileName + ".root";



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
      histVec.push_back("hHits");
      histVec.push_back("hEnergy");
      histVec.push_back("hEnergyDensity");
      histVec.push_back("hLnEnergy");
      histVec.push_back("hLnEnergyDensity");

      shapePlot->PlotEnergyDensityGradient();
      shapePlot->PlotPolar(histVec, false);
      shapePlot->CreateValidationPlot();
      shapePlot->CreateHTML(histVec);
   }
   else
   {
      cout << " Choose .x run_shapepara.C(0) to run ShowerShapeBinning or .x run_shapepara(1) to run ShowerShapePlotting" << endl;
      exit(EXIT_FAILURE);
   }
}
