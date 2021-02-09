/*
  Copyright (C) 2002-2021 CERN for the benefit of the ATLAS collaboration
*/

#include "CLHEP/Random/TRandomEngine.h"

#include "TFile.h"
#include "TTree.h"
#include "TH1F.h"
#include <iostream>
#include <vector>
#include "TGraphErrors.h"
#include <fstream>
#include <string>

using namespace std;

double GetParticleMass(int pid)
{
    if (pid == 11)
        return 0.5;
    else if (pid == 211)
        return 139.6;
    else
        return 0;
}

TCanvas* plotTGraph(TGraphErrors* graph, TGraph* spline)
{


   TGraph* gr = (TGraph*)graph->Clone();

   if (spline) spline->SetLineColor(kBlack);

   gr->SetMarkerColor(kRed);
   gr->SetMarkerSize(1.5);

   TLegend* leg = new TLegend(0.7, 0.55, 0.9, 0.7);
   leg->SetBorderSize(0);
   leg->SetFillStyle(0);
   leg->SetFillColor(0);


   leg->AddEntry(gr, "Geant4", "pe");
   if (spline) leg->AddEntry(spline, "Spline", "l");

   TCanvas * c1 = new TCanvas(gr->GetName(), gr->GetTitle(), 1200, 900);
   c1->cd();
   c1->SetLogx();

   if (spline) spline->Draw("AL");
   gr->Draw("PE");
   leg->Draw();

   ATLASLabel(0.4, 0.2, "Simulation Internal");
   myText(0.2, 0.89, 1, gr->GetTitle());

   gPad->Update();

   return c1;

}

TGraph* GetTGraphSpline(CLHEP::HepRandomEngine *randEngine, TGraph *gr, int pdgid, float etamin, float etamax)
{

   TFCSSimulationState* simulstate = new TFCSSimulationState(randEngine);
   TFCSTruthState* truth = new TFCSTruthState();
   TFCSExtrapolationState* extrapol = new TFCSExtrapolationState();

   TGraph* grdraw = (TGraph*)gr->Clone();
   grdraw->SetMarkerColor(46);
   grdraw->SetMarkerStyle(8);

   TFCSEnergyInterpolationSpline spline("TFCSEnergyInterpolationSpline", "TFCSEnergyInterpolationSpline");
   spline.set_pdgid(pdgid);
   spline.set_Ekin_nominal(0.5 * (grdraw->GetX()[0] + grdraw->GetX()[grdraw->GetN() - 1]));
   spline.set_Ekin_min(grdraw->GetX()[0]);
   spline.set_eta_nominal(0.5 * (etamin + etamax));
   spline.set_eta_min(etamin);
   spline.set_eta_max(etamax);
   spline.InitFromArrayInEkin(gr->GetN(), gr->GetX(), gr->GetY(), "b2e2", 0, 0);
   spline.Print();
   spline.spline().Dump();

   truth->set_pdgid(pdgid);

   TGraph *grSpline = new TGraph();
   grSpline->SetNameTitle("TFCSEnergyInterpolationSplineLogX", "TFCSEnergyInterpolationSpline log x-axis");
   grSpline->GetXaxis()->SetTitle("Ekin [MeV]");
   grSpline->GetYaxis()->SetTitle("<E(reco)>/Ekin(true)");

   int ip = 0;
   for (float Ekin = spline.Ekin_min() * 0.25; Ekin <= spline.Ekin_max() * 4; Ekin *= 1.05 ) {
      truth->SetPxPyPzE(Ekin, 0, 0, Ekin);

      if (spline.simulate(*simulstate, truth, extrapol) != FCSSuccess) {
         std::cout << " Spline is not successful" << std::endl;
         return grSpline;
      }
      grSpline->SetPoint(ip, Ekin, simulstate->E() / Ekin);
      ++ip;
   }

   return grSpline;
}


void runTFCSEnergyInterpolationTGraph(int pid = 22, float etamin = 0., float etamax = 5., bool useFit = false, bool doSpline = true, std::string inputDir = "/eos/atlas/atlascerngroupdisk/proj-simul/InputSamplesSummer18Complete/", std::string plotDir = "plot_Einterpol", std::string ver = "ver03", long seed = 42)
{

   gROOT->SetBatch(kTRUE);
   SetAtlasStyle();

   system(("mkdir -p " + plotDir).c_str());

   std::string method = "Mean";
   if (useFit) method = "Fit";

   CLHEP::TRandomEngine *randEngine = new CLHEP::TRandomEngine();
   randEngine->setSeed(seed);

   TString outputFileName = Form("mc16_13TeV.pid%i.Einterpol%s.%s.root", pid, method.c_str(), ver.c_str());

   std::cout << "Creating parametrisation file " << outputFileName << std::endl;


   TFile *f = new TFile(outputFileName, "RECREATE");

   const int EnergyPoints = 17;

   vector<double> Mean(EnergyPoints);
   vector<double> MeanError(EnergyPoints);
   vector<double> KinEnergies(EnergyPoints);
   vector<double> Momentum(EnergyPoints);
   vector<double> Error(EnergyPoints);
   vector<int> etaEdges;

   // for fit
   int Sigmas[EnergyPoints] = {26, 37, 56, 70, 90, 130, 186, 262, 370, 550, 873, 1373, 2452, 4783, 10880, 22250, 42000};






   
   for (int i = 0; i < EnergyPoints; ++i)
   {
      Momentum[0] = 64;
      if (i != 0){
         Momentum[i] = Momentum[i - 1] * 2;
      }
      KinEnergies[i] = sqrt(pow(Momentum[i],2) + pow(GetParticleMass(pid),2)) - GetParticleMass(pid);
   }


   int neta = (int) (etamax - etamin) / 0.05;

   for (int i = 0; i <= neta; i++)
   {
      etaEdges.push_back(etamin * 100 + i * 5);
   }


   for (int ieta = 0; ieta < etaEdges.size() - 1; ieta++)
   {
      for (int ienergy = 0; ienergy < EnergyPoints; ienergy++)
      {

         TString fileName = inputDir + "*_pid" + (long)pid + "_E" + (long)Momentum[ienergy] + "*_" + (long)etaEdges[ieta] + "_" + (long)etaEdges[ieta + 1] + "*/*";


         float TCE;
         TChain * chain = new TChain("FCS_ParametrizationInput");

         gSystem->Exec(TString("ls ") + fileName + " > $TMPDIR/FCS_ls.$PPID.list");
         TString tmpname = gSystem->Getenv("TMPDIR");
         tmpname += "/FCS_ls.";
         tmpname += gSystem->GetPid();
         tmpname += ".list";

         ifstream infile;
         infile.open(tmpname);
         while (!infile.eof()) {
            string filename;
            getline(infile, filename);
            if (filename != "") {
               chain->Add(filename.c_str(), -1);
            }
         }
         infile.close();

         chain->Add(fileName);
         chain->SetBranchStatus("*", 0);
         chain->SetBranchStatus("total_cell_energy", 1);
         chain->SetBranchAddress("total_cell_energy", &TCE);

         chain->Draw("total_cell_energy>>myh");
         TH1F *myh = (TH1F*)gDirectory->Get("myh");

         if (chain->GetEntries() != myh->GetEntries()) {
            std::cout << "-- Error for Energy " << KinEnergies[ienergy]  << " eta " << etaEdges[ieta] << std::endl;
         }

         if (!useFit) {
            Mean[ienergy] = myh->GetMean() / KinEnergies[ienergy];
            MeanError[ienergy] = myh->GetMeanError() / KinEnergies[ienergy];
         }
         else {
            myh->GetXaxis()->SetRangeUser(10, myh->GetXaxis()->GetBinCenter(100));
            double maximum = myh->GetXaxis()->GetBinCenter(myh->GetMaximumBin());
            double sigma = Sigmas[ienergy];
            if (pid == 211) sigma = sigma * 7;
            double min = maximum - sigma;
            double max = maximum + sigma;
            TF1* g1 = new TF1("g1", "gaus", min, max);
            myh->Fit(g1, "R");

            double mean = g1->GetParameter(1);
            double meanError = g1->GetParError(1);

            Mean[ienergy] = mean / KinEnergies[ienergy];
            MeanError[ienergy] = meanError / KinEnergies[ienergy];
         }

         f->cd();
         TString hName = Form("h_E%.0f_%d_%d", Momentum[ienergy], etaEdges[ieta], etaEdges[ieta + 1]);
         std::cout << "Writing histo " << hName << std::endl;
         myh->Write(hName);

         delete myh;
         delete chain;
      }

      TGraphErrors* graph = new TGraphErrors(EnergyPoints, &KinEnergies[0], &Mean[0], &Error[0], &MeanError[0]);

      TString graphName = Form("Graph_%d_%d", etaEdges[ieta], etaEdges[ieta + 1]);
      TString graphTitle = Form("pid %d, %.2f < |#eta| < %.2f", pid, etaEdges[ieta] * .01, etaEdges[ieta + 1] * .01);

      graph->SetNameTitle(graphName.Data(), graphTitle.Data());
      graph->GetXaxis()->SetTitle("Ekin [MeV]");
      graph->GetYaxis()->SetTitle("<E(reco)>/Ekin(true)");
      graph->Write(graphName);

      TGraph* grSpline = nullptr;

      if (doSpline) grSpline = GetTGraphSpline(randEngine, graph, pid, etaEdges[ieta] * .01, etaEdges[ieta + 1] * .01 );

      TCanvas* c1 = plotTGraph(graph, grSpline);
      TString plotName = Form("%s/pid%d_%s.png", plotDir.c_str(), pid, graphName.Data());
      c1->SaveAs(plotName);

      delete graph;
      delete grSpline;
   }

   f->Close();
   std::cout << "DONE" << std::endl;

   gApplication->Terminate();
}

