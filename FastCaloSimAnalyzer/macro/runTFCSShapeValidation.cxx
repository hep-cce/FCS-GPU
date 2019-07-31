/*
  Copyright (C) 2002-2017 CERN for the benefit of the ATLAS collaboration
*/

// #include "TROOT.h"
// #include "TSystem.h"
#include "TFile.h"
#include "TCanvas.h"
#include "TChain.h"
#include "TGraphErrors.h"
#include "TMath.h"
#include "TTree.h"
#include "TString.h"
#include "TEfficiency.h"
#include "TProfile2D.h"
#include "TPaveText.h"
#include "TLatex.h"
#include "TLegend.h"
#include "TH2.h"

#include <docopt/docopt.h>

#include <iostream>
#include <algorithm>
#include <string>
#include <stdlib.h>
#include <tuple>

#include "ISF_FastCaloSimEvent/TFCSLateralShapeParametrizationHitNumberFromE.h"
#include "ISF_FastCaloSimEvent/TFCSParametrizationChain.h"
#include "ISF_FastCaloSimEvent/TFCSParametrizationEbinChain.h"

#include "FastCaloSimAnalyzer/TFCSAnalyzerBase.h"
#include "FastCaloSimAnalyzer/TFCSAnalyzerHelpers.h"
#include "FastCaloSimAnalyzer/TFCSShapeValidation.h"
#include "FastCaloSimAnalyzer/TFCSValidationEnergy.h"
#include "FastCaloSimAnalyzer/TFCSValidationEnergyAndCells.h"
#include "FastCaloSimAnalyzer/TFCSValidationEnergyAndHits.h"
#include "FastCaloSimAnalyzer/TFCSWriteCellsToTree.h"

#include "TFCSSampleDiscovery.h"

#include <chrono> 

using namespace std;

std::string prefix_E_eta;
std::string prefix_E_eta_title;
std::string prefixlayer;
std::string prefixEbin;
std::string prefixall;
std::string prefixlayer_title;
std::string prefixEbin_title;
std::string prefixall_title;

TFile *fout{};


static const char * USAGE =
    R"(Run toy simulation to validate the shape parametrization

Usage:
  runTFCSShapeValidation [--pdgId <pdgId>] [-s <seed> | --seed <seed>] [-o <file> | --output <file>] [--energy <energy>] [--etaMin <etaMin>] [-l <layer> | --layer <layer>] [--nEvents <nEvents>] [--firstEvent <firstEvent>] [--debug <debug>]
  runTFCSShapeValidation (-h | --help)

Options:
  -h --help                    Show help screen.
  --pdgId <pdgId>              Particle ID [default: 11].
  -s <seed>, --seed <seed>     Random seed [default: 42].
  --energy <energy>            Input sample energy in MeV. Should match energy point on the grid. [default: 65536].
  --etaMin <etaMin>            Minimum eta of the input sample. Should match eta point on the grid. [default: 0.2].
  -o <file>, --output <file>   Output plot file name [default: ShapeValidation.root].
  -l <layer>, --layer <layer>  Layer to analyze [default: 2].
  --nEvents <nEvents>          Number of events to run over with. All events will be used if nEvents<=0 [default: -1].
  --firstEvent <firstEvent>    Run will start from this event [default: 0].
  --debug <debug>              Set debug level to print debug messages [default: 0].
)";


// Sum energies of all cells in a layer
double sumcellE(int analyze_layer, const TFCSSimulationState& simul)
{
  double sumweight = 0;
  for (const auto& iter : simul.cells()) {
    if (iter.first->getSampling() == analyze_layer) sumweight += iter.second;
  }
  return sumweight;
}

void Fill_Evsdxdy_hist(int analyze_layer, const TFCSSimulationState& simul, double sumweight, const TFCSExtrapolationState& extrapol, TProfile* hist1D, TProfile2D* hist2D)
{
  for (const auto& iter : simul.cells()) {
    if (iter.first->getSampling() == analyze_layer) {
      float cell_eta = iter.first->eta();
      float cell_phi = iter.first->phi();

      float TTC_eta = extrapol.eta(analyze_layer, TFCSExtrapolationState::SUBPOS_MID);
      float TTC_phi = extrapol.phi(analyze_layer, TFCSExtrapolationState::SUBPOS_MID);

      float cell_deta = cell_eta - TTC_eta;
      float cell_dphi = TFCSAnalyzerBase::DeltaPhi(cell_phi, TTC_phi);
      float cell_dR = TMath::Sqrt(cell_deta * cell_deta + cell_dphi * cell_dphi);

      TFCSAnalyzerBase::Fill(hist1D, cell_dR, iter.second / sumweight, 1);
      TFCSAnalyzerBase::Fill(hist2D, cell_deta, cell_dphi, iter.second / sumweight, 1);
    }
  }
}

void Fill_Evsdxdy_ratio(int analyze_layer, const TFCSSimulationState& simul1, double sumweight1, const TFCSSimulationState& simul2, double sumweight2, const TFCSExtrapolationState& extrapol, TProfile* hist1D, TProfile2D* hist2D)
{
  for (const auto& iter1 : simul1.cells()) {
    if (iter1.first->getSampling() == analyze_layer) {
      float cell_eta = iter1.first->eta();
      float cell_phi = iter1.first->phi();

      float TTC_eta = extrapol.eta(analyze_layer, TFCSExtrapolationState::SUBPOS_MID);
      float TTC_phi = extrapol.phi(analyze_layer, TFCSExtrapolationState::SUBPOS_MID);

      float cell_deta = cell_eta - TTC_eta;
      float cell_dphi = TFCSAnalyzerBase::DeltaPhi(cell_phi, TTC_phi);
      float cell_dR = TMath::Sqrt(cell_deta * cell_deta + cell_dphi * cell_dphi);

      auto iter2 = simul2.cells().find(iter1.first);
      if (iter2 != simul2.cells().end()) {
        TFCSAnalyzerBase::Fill(hist1D, cell_dR, (iter1.second / sumweight1) / (iter2->second / sumweight2), 1);
        TFCSAnalyzerBase::Fill(hist2D, cell_deta, cell_dphi, (iter1.second / sumweight1) / (iter2->second / sumweight2), 1);
      }
    }
  }
}

TCanvas* Draw_2Dhist(TH1* historg, double zmin = 0.00001, double zmax = 1, bool logz = false, TString name = "", TString title = "")
{
  if (name == "") {
    name = historg->GetName();
    title = historg->GetTitle();
  }
  TH1* hist = (TH1*)historg->Clone(TString("Clone_") + historg->GetName());
  hist->SetTitle(title);
  TCanvas* c = new TCanvas(name, title);
  if (zmin != 0 || zmax != 0) {
    hist->SetMaximum(zmax);
    hist->SetMinimum(zmin);
  }
  hist->SetStats(false);
  hist->Draw("colz");
  c->SetLogz(logz);

  TLatex *t1 = new TLatex();
  t1->SetTextFont(42);
  t1->SetTextSize(0.025);
  t1->SetNDC();
  TString text = Form("Mean x=%5.3f#pm%5.3f y=%5.3f#pm%5.3f", hist->GetMean(1), hist->GetMeanError(1), hist->GetMean(2), hist->GetMeanError(2));
  text += Form(" ; RMS x=%5.3f#pm%5.3f y=%5.3f#pm%5.3f", hist->GetRMS(1), hist->GetRMSError(1), hist->GetRMS(2), hist->GetRMSError(2));
  text += Form(" ; Skew x=%5.3f#pm%5.3f y=%5.3f#pm%5.3f", hist->GetSkewness(1), hist->GetSkewness(11), hist->GetSkewness(2), hist->GetSkewness(12));
  t1->DrawLatex(0.05, 0.01, text);

  if(fout) {
    fout->cd();
    c->Write();
  }  
  c->SaveAs(".png");
  return c;
}

TCanvas* Draw_1Dhist(TH1* hist1, TH1* hist2 = 0, double ymin = 0, double ymax = 0, bool logy = false, TString name = "", TString title = "")
{
  if (name == "") {
    name = hist1->GetName();
    title = hist1->GetTitle();
    if (hist2) {
      name += TString("_") + hist2->GetName();
      title += TString(" AND ") + hist2->GetTitle();
    }
  }
  TCanvas* c = new TCanvas(name, title);

  double min1, max1, rmin1, rmax1;
  TFCSAnalyzerBase::autozoom(hist1, min1, max1, rmin1, rmax1);

  TPaveText *pt = new TPaveText(0.9, 0.5, 1.0, 0.9, "NDC");
  pt->SetFillColor(10);
  pt->SetBorderSize(1);
  TText *t1;

  if (hist2) {
    double min2, max2, rmin2, rmax2;
    TFCSAnalyzerBase::autozoom(hist2, min2, max2, rmin2, rmax2);
    double min = TMath::Min(min1, min2);
    double max = TMath::Max(max1, max2);
    double rmin = TMath::Min(rmin1, rmin2);
    double rmax = TMath::Max(rmax1, rmax2);
    TH1D* newhist1 = TFCSAnalyzerBase::refill(hist1, min, max, rmin, rmax);
    TH1D* newhist2 = TFCSAnalyzerBase::refill(hist2, min, max, rmin, rmax);

    if (newhist2->GetMaximum() > newhist1->GetMaximum()) newhist1->SetMaximum(newhist2->GetMaximum());

    newhist1->SetLineColor(1);
    newhist1->SetFillColor(5);
    newhist1->SetFillStyle(1001);
    newhist1->SetMarkerStyle(2);
    newhist1->SetTitle(title);
    newhist1->SetStats(false);

    if (ymin != 0 || ymax != 0) {
      newhist1->SetMaximum(ymax);
      newhist1->SetMinimum(ymin);
    }

    newhist2->SetMarkerColor(2);
    newhist2->SetLineColor(2);

    TLegend* leg = new TLegend(0.1, 0.005, 0.7, 0.045, "", "NDC");
    leg->SetNColumns(2);

    newhist1->Draw("E2");
    leg->AddEntry(newhist1, hist1->GetTitle(), "lpf");

    newhist2->Draw("same");
    leg->AddEntry(newhist2, hist2->GetTitle(), "lpf");

    leg->Draw();

    t1 = pt->AddText("Mean:");
    t1->SetTextFont(62);
    t1 = pt->AddText(Form("%5.3f#pm%5.3f", hist1->GetMean(), hist1->GetMeanError()));
    t1->SetTextFont(42);
    t1 = pt->AddText(Form("%5.3f#pm%5.3f", hist2->GetMean(), hist2->GetMeanError()));
    t1->SetTextFont(42);
    t1->SetTextColor(2);

    t1 = pt->AddText("RMS:");
    t1->SetTextFont(62);
    t1 = pt->AddText(Form("%5.3f#pm%5.3f", hist1->GetRMS(), hist1->GetRMSError()));
    t1->SetTextFont(42);
    t1 = pt->AddText(Form("%5.3f#pm%5.3f", hist2->GetRMS(), hist2->GetRMSError()));
    t1->SetTextFont(42);
    t1->SetTextColor(2);

    t1 = pt->AddText("Skewness:");
    t1->SetTextFont(62);
    t1 = pt->AddText(Form("%5.3f#pm%5.3f", hist1->GetSkewness(), hist1->GetSkewness(11)));
    t1->SetTextFont(42);
    t1 = pt->AddText(Form("%5.3f#pm%5.3f", hist2->GetSkewness(), hist2->GetSkewness(11)));
    t1->SetTextFont(42);
    t1->SetTextColor(2);
  } else {
    TH1D* newhist1 = TFCSAnalyzerBase::refill(hist1, min1, max1, rmin1, rmax1);
    newhist1->SetLineColor(1);
    newhist1->SetTitle(title);
    newhist1->SetStats(false);
    if (ymin != 0 || ymax != 0) {
      newhist1->SetMaximum(ymax);
      newhist1->SetMinimum(ymin);
    }
    newhist1->Draw("EL");

    t1 = pt->AddText("Mean:");
    t1->SetTextFont(62);
    t1 = pt->AddText(Form("%5.3f#pm%5.3f", hist1->GetMean(), hist1->GetMeanError()));
    t1->SetTextFont(42);

    t1 = pt->AddText("RMS:");
    t1->SetTextFont(62);
    t1 = pt->AddText(Form("%5.3f#pm%5.3f", hist1->GetRMS(), hist1->GetRMSError()));
    t1->SetTextFont(42);

    t1 = pt->AddText("Skewness:");
    t1->SetTextFont(62);
    t1 = pt->AddText(Form("%5.3f#pm%5.3f", hist1->GetSkewness(), hist1->GetSkewness(11)));
    t1->SetTextFont(42);
  }
  pt->Draw();
  c->SetLogy(logy);
  c->SaveAs(".png");
  return c;
}

// Fill energy histograms
void FillEnergyHistos(TH1** hist_E, TFCSShapeValidation *analyze, int analyze_pcabin, TFCSSimulationRun& val1)
{
  hist_E[24] = analyze->InitTH1(prefixEbin + "E_over_Ekintrue_" + val1.GetName(), "1D", 840, 0, 2.0, "E/Ekin(true)", "#");
  hist_E[24]->SetTitle(val1.GetTitle());
  for (int i = 0; i < 24; ++i) {
    hist_E[i] = analyze->InitTH1(prefixEbin + Form("E%02d_over_E_", i) + val1.GetName(), "1D", 840, 0, 1.0, Form("E%d/E", i), "#");
    hist_E[i]->SetTitle(val1.GetTitle());
  }
  for (size_t ievent = 0; ievent < val1.simul().size(); ++ievent) {
    const TFCSSimulationState& simul_val1 = val1.simul()[ievent];
    //cout<<val1.GetName()<<" event="<<ievent<<" E="<<simul_val1.E()<<" Ebin="<<simul_val1.Ebin()<<std::endl;
    if (simul_val1.Ebin() != analyze_pcabin && analyze_pcabin >= 0) continue;
    for (int i = 0; i < 24; ++i) {
      TFCSAnalyzerBase::Fill(hist_E[i], simul_val1.E(i) / simul_val1.E(), 1);
    }
    TFCSAnalyzerBase::Fill(hist_E[24], simul_val1.E() / analyze->get_truthTLV(ievent).Ekin(), 1);
  }
}

// Compare energy histograms
void CompareEnergy(TFCSShapeValidation *analyze, int analyze_pcabin, TFCSSimulationRun& val1, TFCSSimulationRun& val2, TString basename = "", TVirtualPad* summary = 0)
{
  TH1* hist_E_val1[25];
  TH1* hist_E_val2[25];
  FillEnergyHistos(hist_E_val1, analyze, analyze_pcabin, val1);
  FillEnergyHistos(hist_E_val2, analyze, analyze_pcabin, val2);

  TCanvas* c;
  int nlayer = 0;
  double min = 0, max = 0;

  const Int_t nmax = 25;
  TString label[nmax];
  Double_t x1[nmax];
  Double_t y1[nmax];
  Double_t ex1[nmax];
  Double_t ey1[nmax];
  Double_t x2[nmax];
  Double_t y2[nmax];
  Double_t ex2[nmax];
  Double_t ey2[nmax];

  for (int i = 0; i <= 24; ++i) {
    /*
    if(hist_E_val1[i]->GetMean()>0) {
      c=Draw_1Dhist(hist_E_val1[i]);
    }
    if(hist_E_val2[i]->GetMean()>0) {
      c=Draw_1Dhist(hist_E_val2[i]);
    }
    */
    if (hist_E_val1[i]->GetMean() > 0 || hist_E_val2[i]->GetMean() > 0) {
      TString name = basename + "_" + Form("cs%02d_", i) + prefixEbin;
      TString title = basename + ": " + Form("sample=%d, ", i) + prefixEbin_title;
      if (i == 24) {
        name = basename + "_" + Form("total_") + prefixEbin;
        title = basename + ": " + Form("E/E_{true}, ") + prefixEbin_title;
      }
      c = Draw_1Dhist(hist_E_val1[i], hist_E_val2[i], 0, 0, false, name, title);
      if (nlayer == 0) {
        min = TMath::Min(hist_E_val1[i]->GetMean() - hist_E_val1[i]->GetRMS(), hist_E_val2[i]->GetMean() - hist_E_val2[i]->GetRMS());
        max = TMath::Max(hist_E_val1[i]->GetMean() + hist_E_val1[i]->GetRMS(), hist_E_val2[i]->GetMean() + hist_E_val2[i]->GetRMS());
      } else {
        min = TMath::Min(min, TMath::Min(hist_E_val1[i]->GetMean() - hist_E_val1[i]->GetRMS(), hist_E_val2[i]->GetMean() - hist_E_val2[i]->GetRMS()));
        max = TMath::Max(max, TMath::Max(hist_E_val1[i]->GetMean() + hist_E_val1[i]->GetRMS(), hist_E_val2[i]->GetMean() + hist_E_val2[i]->GetRMS()));
      }
      x1[nlayer] = hist_E_val1[i]->GetMean();
      y1[nlayer] = nlayer + 0.15;
      ex1[nlayer] = hist_E_val1[i]->GetRMS();
      ey1[nlayer] = 0;
      x2[nlayer] = hist_E_val2[i]->GetMean();
      y2[nlayer] = nlayer - 0.15;
      ex2[nlayer] = hist_E_val2[i]->GetRMS();
      ey2[nlayer] = 0;
      label[nlayer] = Form("E_{%d} / E", i);
      if (i == 24) label[nlayer] = Form("E/E_{true}");
      ++nlayer;
      if(fout) {
        fout->cd();
        c->Write();
      }  
    }
  }
  if (summary) {
    summary->cd();
    gPad->SetRightMargin(0.01);
    gPad->SetLeftMargin(0.15);
    TString name = basename + "_Summary_" + prefixEbin;
    TString title = basename + ": Summary " + prefixEbin_title;
    double delta = 0.05 * (max - min);
    TH2F* dummy = new TH2F(name + "dummy", title, 100, min - delta, max + delta, nlayer, -0.5, nlayer - 0.5);
    for (int i = 0; i < nlayer; ++i) {
      dummy->GetYaxis()->SetBinLabel(i + 1, label[i]);
    }
    dummy->GetYaxis()->SetTicks("-");
    dummy->GetYaxis()->SetLabelSize(0.07);
    //dummy->GetXaxis()->SetTitle("E/Eref");
    dummy->GetXaxis()->SetLabelSize(0.05);
    dummy->SetStats(false);
    dummy->Draw();
    TGraphErrors* gr1 = new TGraphErrors(nlayer, x1, y1, ex1, ey1);
    gr1->SetName(name);
    gr1->SetTitle(title);
    gr1->SetMarkerColor(1);
    gr1->SetLineColor(1);
    gr1->SetMarkerStyle(1);
    gr1->Draw("Psame");
    TGraphErrors* gr2 = new TGraphErrors(nlayer, x2, y2, ex2, ey2);
    gr2->SetName(name);
    gr2->SetTitle(title);
    gr2->SetMarkerColor(2);
    gr2->SetLineColor(2);
    gr2->SetMarkerStyle(1);
    gr2->Draw("Psame");
  }
}

// Compare validation chain nrval1 with chain nrval2, taking the average of all events instead of event-by-event
void CompareDirectShape2D(TFCSShapeValidation *analyze, int analyze_layer, int analyze_pcabin, TFCSSimulationRun& val1, TFCSSimulationRun& val2)
{
  TProfile* hist_cellEvsdR_val1 = analyze->InitTProfile1D(prefixall + std::string("cellEvsdR_") + val1.GetName(), "1D", 96, 0, 0.3, "dR", "E");
  TProfile* hist_cellEvsdR_val2 = analyze->InitTProfile1D(prefixall + std::string("cellEvsdR_") + val2.GetName(), "1D", 96, 0, 0.3, "dR", "E");
  TProfile* hist_cellEvsdR_directratio = analyze->InitTProfile1D(prefixall + std::string("cellEvsdR_directratio_") + val1.GetName() + "_" + val2.GetName(), "1D", 96, 0, 0.3, "dR", "ratio");

  int nbins_eta = 80;
  double range_eta = 0.25;
  int nbins_phi = 80;
  double range_phi = 10 * TMath::Pi() / 128;
  if (analyze_layer == 1 || analyze_layer == 5) {
    nbins_eta = 160;
    nbins_phi = 20;
  }

  TProfile2D* hist_cellEvsdxdy_val1 = analyze->InitTProfile2D(prefixall + std::string("cellEvsdxdy_") + val1.GetName(), "2D", nbins_eta, -range_eta, range_eta, nbins_phi, -range_phi, range_phi, "deta", "dphi");
  TProfile2D* hist_cellEvsdxdy_val2 = analyze->InitTProfile2D(prefixall + std::string("cellEvsdxdy_") + val2.GetName(), "2D", nbins_eta, -range_eta, range_eta, nbins_phi, -range_phi, range_phi, "deta", "dphi");
  TProfile2D* hist_cellEvsdxdy_directratio = analyze->InitTProfile2D(prefixall + std::string("cellEvsdxdy_directratio_") + val1.GetName() + "_" + val2.GetName(), "2D", nbins_eta, -range_eta, range_eta, nbins_phi, -range_phi, range_phi, "deta", "dphi");

  std::cout << "=============================" << std::endl;
  std::cout << "CompareShape2D: compare '" << val1.GetTitle() << "' with '" << val2.GetTitle() << "'" << std::endl;
  for (size_t ievent = 0; ievent < val1.simul().size(); ++ievent) {
    const TFCSSimulationState& simul_val1 = val1.simul()[ievent];
    if (simul_val1.Ebin() != analyze_pcabin && analyze_pcabin > -1) continue;

    double sumweight_val1 = sumcellE(analyze_layer, simul_val1);
    std::cout << "Event=" << ievent << ": sumweight_val1=" << sumweight_val1 << std::endl;

    Fill_Evsdxdy_hist(analyze_layer, simul_val1, sumweight_val1, analyze->get_extrapol(ievent), hist_cellEvsdR_val1, hist_cellEvsdxdy_val1);
  }
  for (size_t ievent = 0; ievent < val2.simul().size(); ++ievent) {
    const TFCSSimulationState& simul_val2 = val2.simul()[ievent];
    if (simul_val2.Ebin() != analyze_pcabin && analyze_pcabin > -1) continue;

    double sumweight_val2 = sumcellE(analyze_layer, simul_val2);
    std::cout << "Event=" << ievent << ": sumweight_val2=" << sumweight_val2 << std::endl;

    Fill_Evsdxdy_hist(analyze_layer, simul_val2, sumweight_val2, analyze->get_extrapol(ievent), hist_cellEvsdR_val2, hist_cellEvsdxdy_val2);
  }

  hist_cellEvsdR_directratio->Divide(hist_cellEvsdR_val1, hist_cellEvsdR_val2);
  hist_cellEvsdxdy_directratio->Divide(hist_cellEvsdxdy_val1, hist_cellEvsdxdy_val2);

  Draw_2Dhist(hist_cellEvsdxdy_val1, 0.00001, 1, true);
  Draw_2Dhist(hist_cellEvsdxdy_val2, 0.00001, 1, true);
  Draw_2Dhist(hist_cellEvsdxdy_directratio, 1.0 / 3, 1.0 * 3, true);
  Draw_1Dhist(hist_cellEvsdR_val1, hist_cellEvsdR_val2, 0, 0, true);
  Draw_1Dhist(hist_cellEvsdR_directratio);
}

// Compare event-by-event validation chain val1 with chain val2
void CompareShape2D(TFCSShapeValidation *analyze, int analyze_layer, int analyze_pcabin, TFCSSimulationRun& val1, TFCSSimulationRun& val2, TString basename = "")
{
  TProfile* hist_cellEvsdR_val1 = analyze->InitTProfile1D(prefixall + std::string("cellEvsdR_") + val1.GetName(), "1D", 96, 0, 0.3, "dR", "E");
  hist_cellEvsdR_val1->SetTitle(val1.GetTitle());
  TProfile* hist_cellEvsdR_val2 = analyze->InitTProfile1D(prefixall + std::string("cellEvsdR_") + val2.GetName(), "1D", 96, 0, 0.3, "dR", "E");
  hist_cellEvsdR_val2->SetTitle(val2.GetTitle());
  TProfile* hist_cellEvsdR_ratio = analyze->InitTProfile1D(prefixall + std::string("cellEvsdR_ratio_") + val1.GetName() + "_" + val2.GetName(), "1D", 96, 0, 0.3, "dR", "ratio");
  hist_cellEvsdR_ratio->SetTitle(TString(val1.GetTitle()) + " / " + val2.GetTitle());

  int nbins_eta = 80;
  double range_eta = 0.25;
  int nbins_phi = 80;
  double range_phi = 10 * TMath::Pi() / 128;
  if (analyze_layer == 1 || analyze_layer == 5) {
    nbins_eta = 160;
    nbins_phi = 20;
  }

  TProfile2D* hist_cellEvsdxdy_val1 = analyze->InitTProfile2D(prefixall + std::string("cellEvsdxdy_") + val1.GetName(), "2D", nbins_eta, -range_eta, range_eta, nbins_phi, -range_phi, range_phi, "d#eta", "d#phi");
  TProfile2D* hist_cellEvsdxdy_val2 = analyze->InitTProfile2D(prefixall + std::string("cellEvsdxdy_") + val2.GetName(), "2D", nbins_eta, -range_eta, range_eta, nbins_phi, -range_phi, range_phi, "d#eta", "d#phi");
  TProfile2D* hist_cellEvsdxdy_ratio = analyze->InitTProfile2D(prefixall + std::string("cellEvsdxdy_ratio_") + val1.GetName() + "_" + val2.GetName(), "2D", nbins_eta, -range_eta, range_eta, nbins_phi, -range_phi, range_phi, "d#eta", "d#phi");

  std::cout << "=============================" << std::endl;
  std::cout << "CompareShape2D: compare '" << val1.GetTitle() << "' with '" << val2.GetTitle() << "'" << std::endl;
  for (size_t ievent = 0; ievent < TMath::Min(val1.simul().size(), val2.simul().size()); ++ievent) {
    const TFCSSimulationState& simul_val1 = val1.simul()[ievent];
    if (simul_val1.Ebin() != analyze_pcabin && analyze_pcabin >= 0) continue;
    const TFCSSimulationState& simul_val2 = val2.simul()[ievent];
    if (simul_val2.Ebin() != analyze_pcabin && analyze_pcabin >= 0) continue;

    double sumweight_val1 = sumcellE(analyze_layer, simul_val1);
    double sumweight_val2 = sumcellE(analyze_layer, simul_val2);
    //std::cout<<"Event="<<ievent<<": sumweight_val1="<<sumweight_val1<<" , sumweight_val2="<<sumweight_val2<<std::endl;

    Fill_Evsdxdy_hist(analyze_layer, simul_val1, sumweight_val1, analyze->get_extrapol(ievent), hist_cellEvsdR_val1, hist_cellEvsdxdy_val1);
    Fill_Evsdxdy_hist(analyze_layer, simul_val2, sumweight_val2, analyze->get_extrapol(ievent), hist_cellEvsdR_val2, hist_cellEvsdxdy_val2);
    Fill_Evsdxdy_ratio(analyze_layer, simul_val1, sumweight_val1, simul_val2, sumweight_val2, analyze->get_extrapol(ievent), hist_cellEvsdR_ratio, hist_cellEvsdxdy_ratio);
  }

  TCanvas* c;

  TString name;
  TString title;

  name = TString("Shape2D_") + val1.GetName() + "_" + prefixall;
  title = TString(val1.GetTitle()) + ": dE/d#eta/d#phi " + prefixall_title;
  c = Draw_2Dhist(hist_cellEvsdxdy_val1, 0.00001, 1, true, name, title);

  name = TString("Shape2D_") + val2.GetName() + "_" + prefixall;
  title = TString(val2.GetTitle()) + ": dE/d#eta/d#phi " + prefixall_title;
  c = Draw_2Dhist(hist_cellEvsdxdy_val2, 0.00001, 1, true, name, title);

  name = TString("Shape2D_") + basename + "_ratio_" + prefixall;
  title = TString(val1.GetTitle()) + " / " + val2.GetTitle() + ": 2D ratio " + prefixall_title;
  c = Draw_2Dhist(hist_cellEvsdxdy_ratio, 1.0 / 3, 1.0 * 3, true, name, title);

  name = TString("Shape1D_") + basename + "_" + prefixall;
  title = TString(val1.GetTitle()) + " and " + val2.GetTitle() + ": dE/dR " + prefixall_title;
  c = Draw_1Dhist(hist_cellEvsdR_val1, hist_cellEvsdR_val2, 0, 0, true, name, title);
  if(fout) {
    fout->cd();
    c->Write();
  }  

  //name=TString("Shape1D_")+basename+"_ratio_"+prefixall;
  //title=TString(val1.GetTitle())+" / "+val2.GetTitle()+": 1D ratio "+prefixall_title;
  //c=Draw_1Dhist(hist_cellEvsdR_ratio,0,0.5,1.5,false,name,title);
}

void BinLog(TAxis *axis)
{
  //TAxis *axis = h->GetXaxis();
  int bins = axis->GetNbins();

  Axis_t from = TMath::Log(axis->GetXmin());
  Axis_t to = TMath::Log(axis->GetXmax());
  Axis_t width = (to - from) / bins;
  Axis_t *new_bins = new Axis_t[bins + 1];

  for (int i = 0; i <= bins; i++) {
    new_bins[i] = TMath::Exp(from + i * width);
  }
  axis->Set(bins, new_bins);
  delete[] new_bins;
}

void set_prefix(int analyze_layer, int analyze_pcabin)
{
  prefixlayer = prefix_E_eta + Form("cs%02d_", analyze_layer);
  prefixlayer_title = prefix_E_eta_title + Form(", sample=%d", analyze_layer);
  if (analyze_pcabin >= 0) {
    prefixall = prefix_E_eta + Form("cs%02d_pca%d_", analyze_layer, analyze_pcabin);
    prefixall_title = prefix_E_eta_title + Form(", sample=%d, pca=%d", analyze_layer, analyze_pcabin);
    prefixEbin = prefix_E_eta + Form("pca%d_", analyze_pcabin);
    prefixEbin_title = prefix_E_eta_title + Form(", pca=%d", analyze_pcabin);
  } else {
    prefixall = prefix_E_eta + Form("cs%02d_allpca_", analyze_layer);
    prefixall_title = prefix_E_eta_title + Form(", sample=%d, all pca", analyze_layer);
    prefixEbin = prefix_E_eta + Form("allpca_");
    prefixEbin_title = prefix_E_eta_title + Form(", all pca");
  }
}

int runTFCSShapeValidation(int pdgid = 22,
			   int int_E = 65536,
			   double etamin = 0.2,
                           int analyze_layer = 2,
                           const std::string &plotfilename = "ShapeValidation.root",
                           long seed = 42,
			   int nEvents = -1,
			   int firstEvent = 0,
			   int debug = 0)
{

 auto start = std::chrono::system_clock::now();

  FCS::LateralShapeParametrizationArray mapping = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  FCS::init_hit_to_cell_mapping(mapping);

  FCS::LateralShapeParametrizationArray numbersOfHits = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  FCS::init_numbers_of_hits(numbersOfHits, 1);

  /*
  if (pdgid == 22 && analyze_layer == 2) {
    TFCSHitCellMappingWiggle* wigglefunc = new TFCSHitCellMappingWiggle("hit_to_cell_mapping_new", "hit to cell mapping_new");
    TFile* wigglefile = TFile::Open("/eos/atlas/user/a/ahasib/public/Simul-FastCalo/ParametrizationProductionVer02/Wiggle/eta_020_025/wiggle_input_deriv_Sampling_2.ver02.root");
    TH1* wigglehist = (TH1*)wigglefile->Get("pos_deriv_wiggles_Sampling_2");
    float cell_dphi = 0.0245437;
    wigglefunc->initialize(wigglehist, cell_dphi / 2);
    wigglefile->Close();
    hit_to_cell_mapping[2] = wigglefunc;
    hit_to_cell_mapping[2]->set_calosample(analyze_layer);
  }
  */

  TFCSParametrizationBase* fullchain = nullptr;

  std::string paramName = TFCSSampleDiscovery::getParametrizationName();
  auto fullchainfile = std::unique_ptr<TFile>(TFile::Open(paramName.c_str()));
  if (!fullchainfile) {
    std::cerr << "Error: Could not open file '" << paramName << "'" << std::endl;
    return 1;
  }

#ifdef FCS_DEBUG
  fullchainfile->ls();
#endif
  fullchain = dynamic_cast<TFCSParametrizationBase *>(fullchainfile->Get("SelPDGID"));
  fullchainfile->Close();

  double etamax = etamin+0.05;

  std::string particle = "";
  if (pdgid == 22) particle = "photon";
  if (pdgid == 211) particle = "pion";
  if (pdgid == 11) particle = "electron";
  std::string energy = Form("E%d", int_E);
  std::string eta = Form("eta%03d_%03d", TMath::Nint(etamin * 100), TMath::Nint(etamax * 100));

  prefix_E_eta = (particle + "_" + energy + "_" + eta + "_").c_str();
  prefix_E_eta_title = particle + Form(", E=%d MeV, %4.2f<|#eta|<%4.2f", int_E, etamin, etamax);

  std::string energy_label(energy);
  energy_label.erase(0, 1);
  int part_energy = stoi(energy_label);
  std::cout << " energy = " << part_energy << std::endl;

  std::string eta_label(eta);
  eta_label.erase(0, 3);
  std::cout << " eta_label = " << eta_label << std::endl;

  std::string etamin_label = eta_label.substr(0, eta_label.find("_"));
  std::string etamax_label = eta_label.substr(4, eta_label.find("_"));

  auto sample = std::make_unique<TFCSSampleDiscovery>();
  int dsid = sample->findDSID(pdgid, int_E, etamin * 100, 0).dsid;
  FCS::SampleInfo sampleInfo = sample->findSample(dsid);
  
  TString inputSample = sampleInfo.location;
  TString shapefile = sample->getShapeName(dsid);
  TString energyfile = sample->getSecondPCAName(dsid);
  TString pcaSample = sample->getFirstPCAAppName(dsid);
  TString avgSample = sample->getAvgSimShapeName(dsid);

  set_prefix(analyze_layer, -1);
  // gROOT->ProcessLineSync(Form("int analyze_layer=%d", analyze_layer));

#if defined(__linux__)
  std::cout << "* Running on linux system " << std::endl ;
#endif

  std::cout << dsid << "\t" << inputSample << std::endl;

  TChain *inputChain = new TChain("FCS_ParametrizationInput");
  if (inputChain->Add(inputSample, -1) == 0) {
    std::cerr << "Error: Could not open file '" << inputSample << "'" << std::endl;
    return 1;
  }
  // FCS_dsid::wildcard_add_files_to_chain(inputChain,(const char *)inputSample);

  int nentries = inputChain->GetEntries();
  
  if ( nEvents <= 0 ) {
    if ( firstEvent >=0 ) nEvents=nentries;
    else nEvents=nentries;
  }
  else{
    if ( firstEvent >=0 ) nEvents=std::max( 0,std::min(nentries,nEvents+firstEvent) );
    else nEvents = std::max( 0,std::min(nentries,nEvents) );
  }
  
  
  
  std::cout << " * Prepare to run on: " << inputSample << " with entries = " << nentries << std::endl;
  std::cout << " * Running over " << nEvents << " events." << std::endl;
  std::cout << " *   1stPCA file: " << pcaSample << std::endl;
  std::cout << " *   AvgShape file: " << avgSample << std::endl;

  TChain *pcaChain = new TChain("tree_1stPCA");
  pcaChain->Add(pcaSample);
  inputChain->AddFriend("tree_1stPCA");

  // TChain *avgChain = new TChain("AvgShape");
  // avgChain->Add(avgSample);
  // inputChain->AddFriend("AvgShape");
  
  // std::cout << " *   1stPCA: entries = " << pcaChain->GetEntries() << " AvgShape: entries = " << avgChain->GetEntries() << std::endl;

  TFile* fpca = TFile::Open(pcaSample);
  if (!fpca) {
    std::cerr << "Error: Could not open file '" << pcaSample << "'" << std::endl;
    return 1;
  }

  std::vector<int> v_layer;

  TH2I* relevantLayers = (TH2I*)fpca->Get("h_layer");
  int npcabins = relevantLayers->GetNbinsX();
  for (int ibiny = 1; ibiny <= relevantLayers->GetNbinsY(); ibiny++ )
  {
    if ( relevantLayers->GetBinContent(1, ibiny) == 1) v_layer.push_back(ibiny - 1);
  }

  std::cout << " relevantLayers = ";
  for (auto i : v_layer) std::cout << i << " ";
  std::cout << " ; #pca bins = " << npcabins << std::endl;
  
  //////////////////////////////////////////////////////////
  ///// Creat validation steering
  //////////////////////////////////////////////////////////
  TFCSShapeValidation *analyze = new TFCSShapeValidation(inputChain, analyze_layer, seed);
  analyze->set_IsNewSample(true);
//    analyze->set_IsNewSample(false);
  analyze->set_Nentries(nEvents);
  analyze->set_Debug(debug);
  analyze->set_firstevent(firstEvent);
//    analyze->set_nprint(1);

  // gROOT->ProcessLineSync(Form("TFCSShapeValidation* analyze=(TFCSShapeValidation*)%p", analyze));

  std::cout << "=============================" << std::endl;
  //////////////////////////////////////////////////////////
  ///// Chain to read in the energies and cells from the input file
  //////////////////////////////////////////////////////////
  TFCSParametrizationChain* RunOriginal = new TFCSParametrizationChain("original_EnergyAndCells", "original energy and cells from input file");
  TFCSValidationEnergyAndCells* original_EnergyAndCells = new TFCSValidationEnergyAndCells("original_EnergyAndCells", "original energy and cells from input file", analyze);
  original_EnergyAndCells->set_pdgid(pdgid);
  original_EnergyAndCells->set_calosample(analyze_layer);
  original_EnergyAndCells->set_Ekin_bin(-1);
  RunOriginal->push_back(original_EnergyAndCells);
#ifdef FCS_DEBUG
  RunOriginal->Print();
#endif

  int ind_RunOriginal = analyze->add_validation("G4Input", "G4 input", RunOriginal);
  std::cout << "=============================" << std::endl;
  //////////////////////////////////////////////////////////
  ///// Chain to read in the energies from the input file, then simulate the average shape from a histogram
  //////////////////////////////////////////////////////////
  TFCSParametrizationChain* RunOriginalEnergyAvgShapeSim = new TFCSParametrizationChain("original_Energy_sim_avgshape_histo", "original energy from input file, avg shape sim from histo");

  TFCSValidationHitSpy* hitspy_sim1 = new TFCSValidationHitSpy("hitspy1_Energy_sim_shape_histo", "hitspy Nr.1 for original energy from input file, shape sim from histo");
  hitspy_sim1->set_calosample(analyze_layer);

  TFCSValidationHitSpy* hitspy_sim2 = new TFCSValidationHitSpy("hitspy2_Energy_sim_shape_histo", "hitspy Nr.2 for original energy from input file, shape sim from histo");
  hitspy_sim2->set_calosample(analyze_layer);
  hitspy_sim2->set_previous(hitspy_sim1);

  TFCSValidationEnergy* original_Energy = new TFCSValidationEnergy("original_Energy", "original energy from input file", analyze);
  original_Energy->get_layers().push_back(analyze_layer);
  original_Energy->set_n_bins(npcabins);
  original_Energy->set_pdgid(pdgid);

  TFile* file_avgshape=nullptr;
  TTree* tree_avgshape=nullptr;
  
  cout<<"Check avg shape file"<<endl;

  bool use_avg_shape_file=false;
  if( use_avg_shape_file) {
    // Load average simulated shape from file
    TFile* avgshapefile = TFile::Open(avgSample);
    if (!avgshapefile) {
      std::cerr << "Error: Could not open file '" << avgSample << "'" << std::endl;
      return 1;
    }
#ifdef FCS_DEBUG
    avgshapefile->ls();
#endif
    
    cout<<"Get: "<<prefix_E_eta + Form("csall_allpca_hist_hitspy1_sample%d_geodphi_1D",analyze_layer)<<endl;
    
    hitspy_sim1->hist_hitgeo_dphi() = (TH1*)avgshapefile->Get((prefix_E_eta + Form("csall_allpca_hist_hitspy1_sample%d_geodphi_1D",analyze_layer)).c_str());
    hitspy_sim1->hist_hitgeo_dphi()->SetDirectory(0);
    hitspy_sim1->hist_hitgeo_matchprevious_dphi() = (TH1*)avgshapefile->Get((prefix_E_eta + Form("csall_allpca_hist_hitspy1_sample%d_geomatchprevious_dphi_1D",analyze_layer)).c_str());
    hitspy_sim1->hist_hitgeo_matchprevious_dphi()->SetDirectory(0);
    hitspy_sim2->hist_hitgeo_dphi() = (TH1*)avgshapefile->Get((prefix_E_eta + Form("csall_allpca_hist_hitspy2_sample%d_geodphi_1D",analyze_layer)).c_str());
    hitspy_sim2->hist_hitgeo_dphi()->SetDirectory(0);
    hitspy_sim2->hist_hitgeo_matchprevious_dphi() = (TH1*)avgshapefile->Get((prefix_E_eta + Form("csall_allpca_hist_hitspy2_sample%d_geomatchprevious_dphi_1D",analyze_layer)).c_str());
    hitspy_sim2->hist_hitgeo_matchprevious_dphi()->SetDirectory(0);
    avgshapefile->Close();

    TFCSValidationEnergyAndCells* original_EnergyAndAvgCells = new TFCSValidationEnergyAndCells("original_EnergyAndAvgCells", "original energy from input file and average cell from stored simulation", analyze);
    original_EnergyAndAvgCells->set_pdgid(pdgid);
    original_EnergyAndAvgCells->set_calosample(analyze_layer);
    original_EnergyAndAvgCells->set_Ekin_bin(-1);
    original_EnergyAndAvgCells->set_UseAvgShape();
    RunOriginalEnergyAvgShapeSim->push_back(original_EnergyAndAvgCells);
  } else {
    // Simulate average shape
    RunOriginalEnergyAvgShapeSim->push_back(original_Energy);

    hitspy_sim1->hist_hitgeo_dphi() = analyze->InitTH1(prefixall + "hist_hitspy_sim1_geodphi", "1D", 256, -TMath::Pi() / 64, TMath::Pi() / 64, "dphi", "#hits");
    hitspy_sim1->hist_hitgeo_matchprevious_dphi() = analyze->InitTH1(prefixall + "hist_hitspy_sim1_geomatchprevious_dphi", "1D", 256, -TMath::Pi() / 64, TMath::Pi() / 64, "dphi", "#hits");

    hitspy_sim2->hist_hitgeo_dphi() = analyze->InitTH1(prefixall + "hist_hitspy_sim2_geodphi", "1D", 256, -TMath::Pi() / 64, TMath::Pi() / 64, "dphi", "#hits");
    hitspy_sim2->hist_hitgeo_matchprevious_dphi() = analyze->InitTH1(prefixall + "hist_hitspy_sim2_geomatchprevious_dphi", "1D", 256, -TMath::Pi() / 64, TMath::Pi() / 64, "dphi", "#hits");

    TFCSParametrizationEbinChain* EbinChainAvgShape = FCS::NewShapeEbinCaloSampleChain(original_Energy, mapping, {}, shapefile.Data(), pdgid, int_E, etamin, etamax);
    RunOriginalEnergyAvgShapeSim->push_back(EbinChainAvgShape);
    for (size_t i = 0; i < EbinChainAvgShape->size(); ++i) {
      if ((*EbinChainAvgShape)[i]->InheritsFrom(TFCSLateralShapeParametrizationHitChain::Class())) {
        TFCSLateralShapeParametrizationHitChain* hitchain = (TFCSLateralShapeParametrizationHitChain*)(*EbinChainAvgShape)[i];
        if (hitchain->size() > 0) {
          auto it = hitchain->chain().begin() + 1;
          hitchain->chain().insert(it, hitspy_sim1);
        }
        hitchain->push_back(hitspy_sim2);
      }
    }
    
    file_avgshape=TFile::Open("OutputAvgShape.root","RECREATE");
    if (!file_avgshape) {
      std::cerr << "Error: Could not create file '" << "OutputAvgShape.root" << "'" << std::endl;
      return 1;
    }

    tree_avgshape=nullptr;
    if(file_avgshape) {
      tree_avgshape=new TTree(Form("AvgShape"),Form("AvgShape"));
      TFCSWriteCellsToTree* tree_writer_AvgShapeSim=new TFCSWriteCellsToTree("tree_writer_AvgShapeSim","Tree writer for original energy from input file, shape sim from histo",tree_avgshape);
      RunOriginalEnergyAvgShapeSim->push_back(tree_writer_AvgShapeSim);
      file_avgshape->Add(hitspy_sim1->hist_hitgeo_dphi());
      file_avgshape->Add(hitspy_sim1->hist_hitgeo_matchprevious_dphi());
      file_avgshape->Add(hitspy_sim2->hist_hitgeo_dphi());
      file_avgshape->Add(hitspy_sim2->hist_hitgeo_matchprevious_dphi());
    }
    // TODO: why is this needed?
    // gROOT->cd();
  }

#ifdef FCS_DEBUG
  RunOriginalEnergyAvgShapeSim->Print();
#endif
  
  int ind_RunOriginalEnergyAvgShapeSim = -1;
  ind_RunOriginalEnergyAvgShapeSim = analyze->add_validation("AvgShape", "Average shape sim", RunOriginalEnergyAvgShapeSim);
  std::cout << "=============================" << std::endl;
  //////////////////////////////////////////////////////////
  ///// Chain to read in the energies from the input file, then simulate the shape from a histogram
  //////////////////////////////////////////////////////////
  TFCSParametrizationChain* RunOriginalEnergyShapeSim = new TFCSParametrizationChain("original_Energy_sim_shape_histo", "original energy from input file, shape sim from histo");
  RunOriginalEnergyShapeSim->push_back(original_Energy);

  if(analyze_layer<8) {
    numbersOfHits[analyze_layer] = new TFCSLateralShapeParametrizationHitNumberFromE("numbers_of_hits_EM", "Calc numbers of hits EM", 0.101, 0.002);
    numbersOfHits[analyze_layer]->set_calosample(analyze_layer);
  }  

  TFCSParametrizationEbinChain* EbinChain = FCS::NewShapeEbinCaloSampleChain(original_Energy, mapping, numbersOfHits, shapefile.Data(), pdgid, int_E, etamin, etamax);
  RunOriginalEnergyShapeSim->push_back(EbinChain);

#ifdef FCS_DEBUG
  RunOriginalEnergyShapeSim->Print();
#endif

  int ind_RunOriginalEnergyShapeSim = analyze->add_validation("Shape", "Shape sim", RunOriginalEnergyShapeSim);
  std::cout << "=============================" << std::endl;
  //////////////////////////////////////////////////////////
  ///// Chain to simulate energy from PCS and the shape from a histogram
  //////////////////////////////////////////////////////////
  int ind_fullchain = -1;
  if (fullchain) {
    ind_fullchain = analyze->add_validation("AllSim", "Energy+shape sim", fullchain);
    std::cout << "=============================" << std::endl;
  }
  //////////////////////////////////////////////////////////
  ///// Chain to read in the energies and hits from the input file
  //////////////////////////////////////////////////////////
  TFCSParametrizationChain* RunOriginalHits = new TFCSParametrizationChain("original_EnergyAndHits", "original energy and hits from input file");
  TFCSValidationEnergyAndHits* original_EnergyAndHits = new TFCSValidationEnergyAndHits("original_EnergyAndHits", "original energy and hits from input file", analyze);
  original_EnergyAndHits->set_pdgid(pdgid);
  original_EnergyAndHits->set_calosample(analyze_layer);
  original_EnergyAndHits->set_Ekin_bin(-1);
  TFCSValidationHitSpy* hitspy_org = new TFCSValidationHitSpy("hitspy_original_EnergyAndHits", "hitspy for original energy and hits from input file");
  hitspy_org->set_calosample(analyze_layer);
  hitspy_org->set_previous(&original_EnergyAndHits->get_hitspy());
  hitspy_org->hist_hitgeo_dphi() = analyze->InitTH1(prefixall + "hist_hitspy_org_geodphi", "1D", 256, -TMath::Pi() / 64, TMath::Pi() / 64, "dphi", "#hits");
  hitspy_org->hist_hitgeo_matchprevious_dphi() = analyze->InitTH1(prefixall + "hist_hitspy_org_geomatchprevious_dphi", "1D", 256, -TMath::Pi() / 64, TMath::Pi() / 64, "dphi", "#hits");
  original_EnergyAndHits->push_back(hitspy_org);
  original_EnergyAndHits->push_back(mapping[analyze_layer]);

  TH1* hist_cellEvsSumHitE = analyze->InitTH2(prefixall + "cellEvsSumHitE", "2D", 101, -20, 2000, 101, -20, 2000, "SumHitE/MeV", "CellE/MeV");
  TH1* hist_cellEvsSumHitEprofile = analyze->InitTProfile1D(prefixall + "cellEvsSumHitEprofile", "2D", 200, 0, 40, "SumHitE/MeV", "CellE/MeV", "");
  TH1* hist_cellEvsSumHitE_largerange = analyze->InitTH2(prefixall + "cellEvsSumHitE_largerange", "2D", 1000, 0, 20000, 1000, 0, 20000, "SumHitE/MeV", "CellE/MeV");
  TH1* hist_cellEvsSumHitEprofile_largerange = analyze->InitTProfile1D(prefixall + "cellEvsSumHitEprofile_largerange", "2D", 1000, 0, 20000, "SumHitE/MeV", "CellE/MeV", "");
  original_EnergyAndHits->add_histo(hist_cellEvsSumHitE);
  original_EnergyAndHits->add_histo(hist_cellEvsSumHitEprofile);
  original_EnergyAndHits->add_histo(hist_cellEvsSumHitE_largerange);
  original_EnergyAndHits->add_histo(hist_cellEvsSumHitEprofile_largerange);

  original_EnergyAndHits->hist_hit_time() = analyze->InitTH1(prefixall + "hist_hit_time", "1D", 150, -100, 200, "time", "energy");
  original_EnergyAndHits->hist_problematic_hit_eta_phi() = analyze->InitTH2(prefixall + "problematic_hit_eta_phi", "2D", 240, -3, +3, 256, -TMath::Pi(), TMath::Pi(), "eta", "phi");


  for (int tcut = 5; tcut <= 55; tcut += 25) {
    original_EnergyAndHits->m_hist_ratioErecoEhit_vs_Ehit_starttime.push_back(-5);
    original_EnergyAndHits->m_hist_ratioErecoEhit_vs_Ehit_endtime.push_back(tcut);
    TH2* hist = analyze->InitTH2(prefixall + Form("cellEratiovsSumHitE_largerange_t%d", tcut), "2D", 1000, 0.5, 20000, 160, 0.8, 1.60, "SumHitE/MeV", "CellE/SumHitE");
    BinLog(hist->GetXaxis());
    original_EnergyAndHits->m_hist_ratioErecoEhit_vs_Ehit.push_back(hist);
    hist = analyze->InitTH2(prefixall + Form("cellEratiovsSumG4HitE_largerange_t%d", tcut), "2D", 1000, 0.5, 20000, 160, 0.8, 1.60, "SumG4HitE/MeV", "CellE/SumHitE");
    BinLog(hist->GetXaxis());
    original_EnergyAndHits->m_hist_ratioErecoEG4hit_vs_EG4hit.push_back(hist);
  }

  RunOriginalHits->push_back(original_EnergyAndHits);
  
#ifdef FCS_DEBUG
  RunOriginalHits->Print();
#endif

  int ind_RunOriginalHits=-1;
  ind_RunOriginalHits=analyze->add_validation("G4Hits","G4 hits",RunOriginalHits);
  std::cout << "=============================" << std::endl;


 auto t1 = std::chrono::system_clock::now();
std::chrono::duration<double> t_before = t1-start;

  //////////////////////////////////////////////////////////

  //////////////////////////////////////////////////////////
  ///// Run over events
  //////////////////////////////////////////////////////////

  analyze->LoopEvents(-1);

 auto t2 = std::chrono::system_clock::now();
std::chrono::duration<double> t_loop = t2-t1;

  if(file_avgshape) {
    std::cout << "= Average Shape output tree =" << std::endl;
    file_avgshape->Write();
#ifdef FCS_DEBUG
    file_avgshape->ls();
    //tree_avgshape->Print();
#endif
    std::cout << "=============================" << std::endl;
  }  

  // return;

  if(plotfilename!="") { 
    fout = TFile::Open(plotfilename.c_str(), "recreate");
    if (!fout) {
      std::cerr << "Error: Could not create file '" << plotfilename << "'" << std::endl;
      return 1;
    }
    fout->cd();
  }  

  TCanvas* c;
  int nbinx = TMath::CeilNint(TMath::Sqrt(npcabins + 1));
  int nbiny = TMath::CeilNint((npcabins + 1.0) / nbinx);
  int ibin = 1;
  TString nameenergy = "Energy";
  TString nameavgshape = "G4AvgShape";
  TString nameshape = "ShapeAvgShape";
  TCanvas* cenergy = 0;
  if(ind_fullchain >= 0) {
    cenergy = new TCanvas(nameenergy + "_summary_" + prefix_E_eta, prefix_E_eta + ": " + nameenergy + " Summary", 1600, 1200);
    cenergy->Divide(nbinx, nbiny);
  }  
  //for(int i=-1;i<=0;++i) {
  for (int i = -1; i <= npcabins; ++i) {
    if (i == 0) continue;
    int analyze_pcabin = i;
    set_prefix(analyze_layer, analyze_pcabin);
    if (ind_RunOriginal < 0) continue;

    TFCSSimulationRun& val1 = analyze->validations()[ind_RunOriginal];
    if (ind_fullchain >= 0) {
      TFCSSimulationRun& val2 = analyze->validations()[ind_fullchain];
      CompareEnergy(analyze, analyze_pcabin, val1, val2, nameenergy, cenergy->cd(ibin));
    }
    TH1* ratio_input_avgsim = 0;
    if (ind_RunOriginalEnergyAvgShapeSim >= 0) {
      TFCSSimulationRun& val2 = analyze->validations()[ind_RunOriginalEnergyAvgShapeSim];
      CompareShape2D(analyze, analyze_layer, analyze_pcabin, val1, val2, nameavgshape);
      cout << "key=" << prefixall + std::string("cellEvsdR_ratio_") + val1.GetName() + "_" + val2.GetName() + "_1D" << endl;
      ratio_input_avgsim = analyze->histMap()[prefixall + std::string("cellEvsdR_ratio_") + val1.GetName() + "_" + val2.GetName() + "_1D"];

      TH1* ratio_sim_avgsim = 0;
      if (ind_RunOriginalEnergyShapeSim >= 0) {
        TFCSSimulationRun& val1 = analyze->validations()[ind_RunOriginalEnergyShapeSim];
        CompareShape2D(analyze, analyze_layer, analyze_pcabin, val1, val2, nameshape);
        cout << "key=" << prefixall + std::string("cellEvsdR_ratio_") + val1.GetName() + "_" + val2.GetName() + "_1D" << endl;
        ratio_sim_avgsim = analyze->histMap()[prefixall + std::string("cellEvsdR_ratio_") + val1.GetName() + "_" + val2.GetName() + "_1D"];

        cout << "ratio_input_avgsim=" << ratio_input_avgsim << " ratio_sim_avgsim=" << ratio_sim_avgsim << endl;
        if (ratio_input_avgsim && ratio_sim_avgsim) {
          TString name = TString("Shape1D_ratio_") + prefixall;
          TString title = TString("(") + ratio_input_avgsim->GetTitle() + ") / (" + ratio_sim_avgsim->GetTitle() + ") : " + prefixall_title;
          c = Draw_1Dhist(ratio_input_avgsim, ratio_sim_avgsim, 0.5, 1.5, false, name, title);
          if(fout) {
            fout->cd();
            c->Write();
          }  
        }

      }
    }

    ++ibin;
  }
  if(cenergy) {
    if(fout) {
      fout->cd();
      cenergy->Write();
    }  
    cenergy->SaveAs(".png");
  }  

  //CompareDirectShape2D(analyze, analyze_layer, analyze_pcabin, analyze->validations()[ind_RunOriginal], analyze->validations()[ind_fullchain]);

  //CompareDirectShape2D(analyze,analyze_layer,analyze_pcabin,analyze->validations()[ind_RunOriginal],analyze->validations()[2]);

  //CompareShape2D(analyze,analyze_layer,analyze_pcabin,analyze->validations()[0],analyze->validations()[1]);

  //CompareShape2D(analyze,analyze_layer,analyze_pcabin,analyze->validations()[2],analyze->validations()[1]);

  //CompareShape2D(analyze,analyze_layer,analyze_pcabin,0,4);

  //c=Draw_2Dhist(hist_cellEvsSumHitE,0,1000);
  //hist_cellEvsSumHitEprofile->SetLineColor(2);
  //hist_cellEvsSumHitEprofile->Draw("sames");

  //c=Draw_2Dhist(hist_cellEvsSumHitE_largerange,0,1000);
  //hist_cellEvsSumHitEprofile_largerange->SetLineColor(2);
  //hist_cellEvsSumHitEprofile_largerange->Draw("sames");

  //c=Draw_2Dhist(original_EnergyAndHits->hist_problematic_hit_eta_phi(),0,10);

  //c=Draw_1Dhist(original_EnergyAndHits->hist_hit_time(),0,true);
  /*
  for(auto hist:original_EnergyAndHits->m_hist_ratioErecoEhit_vs_Ehit) {
    c=Draw_2Dhist(hist,0,100);
    c->SetLogx();
  }
  for(auto hist:original_EnergyAndHits->m_hist_ratioErecoEG4hit_vs_EG4hit) {
    c=Draw_2Dhist(hist,0,100);
    c->SetLogx();
  }
  */


  if(ind_RunOriginalHits>=0 && ind_RunOriginalEnergyAvgShapeSim>=0) {
    c = Draw_1Dhist(hitspy_org->hist_hitgeo_dphi(), hitspy_sim2->hist_hitgeo_dphi());
    c = Draw_1Dhist(hitspy_org->hist_hitgeo_matchprevious_dphi(), hitspy_sim2->hist_hitgeo_matchprevious_dphi());

    TEfficiency* eff_org = new TEfficiency(*hitspy_org->hist_hitgeo_matchprevious_dphi(), *hitspy_org->hist_hitgeo_dphi());
    TEfficiency* eff_sim2 = new TEfficiency(*hitspy_sim2->hist_hitgeo_matchprevious_dphi(), *hitspy_sim2->hist_hitgeo_dphi());
    eff_sim2->SetLineColor(2);

    c = new TCanvas(TString("efficiency_") + hitspy_org->hist_hitgeo_matchprevious_dphi()->GetName(), TString("Efficiency for ") + hitspy_org->hist_hitgeo_matchprevious_dphi()->GetTitle());
    eff_org->Draw();
    eff_sim2->Draw("same");
    if(fout) {
      fout->cd();
      c->Write();
    }  
    c->SaveAs(".png");
  }  

  /*
  c=new TCanvas((prefixall+"_ratio_comp").c_str(),"Comparison ratios");
  analyze->histMap()[prefixall+"cellEvsdR_ratio_original_EnergyAndCells_original_Energy_sim_avgshape_histo_1D"]->DrawClone("E3L");
  analyze->histMap()[prefixall+"cellEvsdR_ratio_original_Energy_sim_shape_histo_original_Energy_sim_avgshape_histo_1D"]->SetLineColor(2);
  analyze->histMap()[prefixall+"cellEvsdR_ratio_original_Energy_sim_shape_histo_original_Energy_sim_avgshape_histo_1D"]->DrawClone("same");
  c->SaveAs(".png");
  */

  if(file_avgshape) {
    file_avgshape->Close();
    file_avgshape=nullptr;
  }  
  if(fout) {
#ifdef FCS_DEBUG
    fout->ls();
#endif
    fout->Close(); //Close will delete all histograms, so no interactive change possible afterwards
    delete fout;
  }
 auto t3 = std::chrono::system_clock::now();
std::chrono::duration<double> t_after = t3-t2;

 std::cout <<  "Time before eventloop :" << t_before.count() <<" s" << std::endl ;
 std::cout <<  "Time eventloop :" << t_loop.count() <<" s" << std::endl ;
 std::cout <<  "Time after eventloop :" << t_after.count() <<" s" << std::endl ;

  return 0;
}

int main(int argc, char **argv)
{
  std::map<std::string, docopt::value> args
    = docopt::docopt(USAGE, {argv + 1, argv + argc}, true);

  int pdgId = args["--pdgId"].asLong();
  int energy = args["--energy"].asLong();
  double etamin = std::stof( args["--etaMin"].asString() );
  long seed = args["--seed"].asLong();
  std::string output = args["--output"].asString();
  int layer = args["--layer"].asLong();
  int nEvents = args["--nEvents"].asLong();
  int firstEvent = args["--firstEvent"].asLong();
  int debug = args["--debug"].asLong();

  return runTFCSShapeValidation(pdgId, energy, etamin, layer, output, seed,nEvents,firstEvent,debug);
}
