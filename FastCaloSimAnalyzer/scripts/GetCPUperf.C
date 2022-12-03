/*
  Copyright (C) 2002-2023 CERN for the benefit of the ATLAS collaboration
*/

#include "TROOT.h"
// #include "TH1.h"
// #include "TGraph.h"
// #include "TLegend.h"

#include "../atlasstyle/AtlasStyle.C"


#ifdef __CLING__
// these are not headers - do not treat them as such - needed for ROOT6
#include "../atlasstyle/AtlasLabels.C"
#include "../atlasstyle/AtlasUtils.C"
#endif


void GetCPUperf(std::string particle) {

    SetAtlasStyle();


    string label = "#gamma , 0.20 < |#eta| < 0.25";
    if (particle == "pion")
        label = "#pi^{#pm} , 0.20 < |#eta| < 0.25";


    std::vector<int> v_energy{8, 65, 262};
    std::vector<int> v_eta{20};
    std::vector<std::string> v_simul{"G4", "FCSV2", "AF2"};


    std::string topDir = "/eos/atlas/user/a/ahasib/public/Simul-FastCalo/CPUperformance/rel_21_0_73";

    for (auto eta : v_eta) {

        TMultiGraph *mg = new TMultiGraph();

        TCanvas* c1 = new TCanvas(Form("%s_eta%i:", particle.c_str(), eta), Form("%s_eta%i:", particle.c_str(), eta), 0, 0, 1200, 900);

        c1->cd();

        TLegend* leg = new TLegend(0.18, 0.65, 0.32, 0.85);
        leg->SetBorderSize(0);
        leg->SetFillStyle(0);
        leg->SetFillColor(0);

        for (int isimul = 0; isimul < v_simul.size(); isimul++) {

            std::string simul = v_simul.at(isimul);

            int n = v_simul.size();
            float x[n];
            float ex[n];
            float y[n];
            float ey[n];

            for (int i = 0; i < v_energy.size(); i++) {

                int energy = v_energy.at(i);

                TString file = Form("%s/%s_%iGeV_eta%i_%s.root", topDir.c_str(), particle.c_str(), energy, eta, simul.c_str());


                std::cout << file << std::endl;

                TFile* f = TFile::Open(file);
                TH1F* h_cpu = new TH1F();
                // TH1::StatOverflows(kTRUE);
                // f->GetObject("000/avg_cpu_PerfMonSlice.000;1", h_cpu);
                f->GetObject("000/cpu_PerfMonSlice.000;1", h_cpu);
                // h_cpu->GetXaxis()->SetRange(3, 90);

                TH1F* havg = new TH1F("havg", "havg", 900000, 0, 90000);
                havg->Sumw2();
                // std::cout << "Energy " << energy << " GeV:" << h_cpu->GetMean() << " +/- " << h_cpu->GetMeanError() << std::endl;

                for (int i = 2; i < h_cpu->GetNbinsX(); i++)
                    havg->Fill(h_cpu->GetBinContent(i));


                float mean = havg->GetMean();
                float rms = havg->GetRMS();

                std::cout << "Energy " << energy << " GeV:" << mean << " +/- " << rms << std::endl;


                x[i] = (float)energy;
                ex[i] = 0.;
                y[i] = mean;
                ey[i] = rms;

                delete h_cpu;
                delete havg;
                f->Close();

            }

            TGraphErrors* gr = new TGraphErrors(n, x, y, ex, ey);



            std::string flavor = "";

            if (simul == "G4" ) {
                flavor = "Geant4";
                gr->SetMarkerColor(kBlack);
                gr->SetLineColor(kBlack);
                gr->SetLineWidth(2);
            }
            if (simul == "AF2") {
                flavor = "AF2";
                gr->SetMarkerColor(kBlue);
                gr->SetMarkerSize(2);
                gr->SetLineColor(kBlue);
                gr->SetLineWidth(2);
                gr->SetLineStyle(2);
                gr->SetMarkerStyle(kOpenCircle);
            }
            if (simul == "FCSV2") {
                flavor = "FCSV2";
                gr->SetMarkerColor(kRed);
                gr->SetMarkerSize(2);
                gr->SetLineColor(kRed);
                gr->SetLineWidth(2);

            }

            leg->AddEntry(gr, flavor.c_str(), "lpe");


            gr->SetMarkerSize(2);
            gr->GetXaxis()->SetTitle("Energy [GeV]");
            gr->GetYaxis()->SetTitle("CPU / evt. <ms>");


            mg->Add(gr);


            // if (isimul == 0) gr->Draw("apl");
            // else gr->Draw("pl same");
            // leg->Draw();
        }

        mg->Draw("alpe1");

        leg->Draw();


        c1->SetLogy();
        c1->SetLogx();

        mg->GetXaxis()->SetLimits(5, 500);


        mg->GetXaxis()->SetTitle("Energy [GeV]");
        mg->GetYaxis()->SetTitle("Average CPU time / evt. [ms]");

        ATLASLabel(0.18, 0.05, "Simulation Internal");
        myText(0.18, 0.9, 1, label.c_str());

        c1->Modified();
        c1->Update();

        c1->SaveAs((particle + "_CPU_performance.pdf").c_str());

    }




}
