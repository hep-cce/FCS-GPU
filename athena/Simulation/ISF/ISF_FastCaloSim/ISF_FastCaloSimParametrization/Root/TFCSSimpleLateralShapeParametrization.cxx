/*
  Copyright (C) 2002-2018 CERN for the benefit of the ATLAS collaboration
*/

#include "CLHEP/Random/RandGauss.h"

#include "ISF_FastCaloSimParametrization/TFCSSimpleLateralShapeParametrization.h"
#include "ISF_FastCaloSimEvent/TFCSExtrapolationState.h"
#include "ISF_FastCaloSimEvent/TFCSSimulationState.h"
#include "ISF_FastCaloSimEvent/FastCaloSim_CaloCell_ID.h"

#include "TMath.h"

//=============================================
//======= TFCSLateralShapeParametrization =========
//=============================================

TFCSSimpleLateralShapeParametrization::TFCSSimpleLateralShapeParametrization(
    const char *name, const char *title)
    : TFCSLateralShapeParametrizationHitBase(name, title) {
  m_sigmaX = 0;
  m_sigmaY = 0;
}

FCSReturnCode TFCSSimpleLateralShapeParametrization::simulate_hit(
    Hit &hit, TFCSSimulationState &simulstate, const TFCSTruthState * /*truth*/,
    const TFCSExtrapolationState *extrapol) {
  if (!simulstate.randomEngine()) {
    return FCSFatal;
  }

  int cs = calosample();
  hit.eta() = 0.5 * (extrapol->eta(cs, CaloSubPos::SUBPOS_ENT) +
                     extrapol->eta(cs, CaloSubPos::SUBPOS_EXT));
  hit.phi() = 0.5 * (extrapol->phi(cs, CaloSubPos::SUBPOS_ENT) +
                     extrapol->phi(cs, CaloSubPos::SUBPOS_EXT));
  hit.E() *= 1;

  double x, y;
  getHitXY(simulstate.randomEngine(), x, y);

  // delta_eta and delta_phi;
  double delta_eta = x;
  double delta_phi = y;

  hit.eta() += delta_eta;
  hit.phi() += delta_phi;

  return FCSSuccess;
}

bool TFCSSimpleLateralShapeParametrization::Initialize(float input_sigma_x,
                                                       float input_sigma_y) {
  m_sigmaX = input_sigma_x;
  m_sigmaY = input_sigma_y;
  return true;
}

bool TFCSSimpleLateralShapeParametrization::Initialize(const char *filepath,
                                                       const char *histname) {
  // input file with histogram to fit
  TFile *f = new TFile(filepath);
  if (f == NULL) return false;

  // histogram with hit pattern
  TH2D *inputShape = (TH2D *)f->Get(histname);
  if (inputShape == NULL) return false;

  // Function to fit with
  double hiEdge =
      inputShape->GetYaxis()->GetBinLowEdge(inputShape->GetNbinsY());
  TF1 *x_func = new TF1("fx", "gaus", -hiEdge, hiEdge);
  TF1 *y_func = new TF1("fy", "gaus", -hiEdge, hiEdge);

  // Project into x and y histograms
  TH1F *h_xrms = new TH1F("h_xrms", "h_xrms", 100, -hiEdge, hiEdge);
  TH1F *h_yrms = new TH1F("h_yrms", "h_yrms", 100, -hiEdge, hiEdge);

  double val = 0;  // bin content
  double r = 0;    // radius
  double a = 0;    // angle
  double ypos = 0;
  double xpos = 0;

  // Loop over to project along axes, takes bin center as position
  for (int xbin = 1; xbin < inputShape->GetNbinsX() + 1; xbin++) {
    a = inputShape->GetXaxis()->GetBinCenter(xbin);

    for (int ybin = 1; ybin < inputShape->GetNbinsY() + 1; ybin++) {
      val = inputShape->GetBinContent(xbin, ybin);

      r = inputShape->GetYaxis()->GetBinCenter(ybin);

      ypos = r * TMath::Sin(a);
      xpos = r * TMath::Cos(a);

      h_xrms->Fill(xpos, val);
      h_yrms->Fill(ypos, val);
    }
  }

  h_xrms->Fit(x_func, "0");
  TF1 *fitx = h_xrms->GetFunction("fx");
  // posibly center

  h_yrms->Fit(y_func, "0");
  TF1 *fity = h_yrms->GetFunction("fy");
  // posibly center

  // Finally set sigma
  m_sigmaX = fitx->GetParameter(2);
  m_sigmaY = fity->GetParameter(2);

  // clean up
  delete x_func;
  delete y_func;

  delete h_xrms;
  delete h_yrms;
  f->Close();

  return true;
}

void TFCSSimpleLateralShapeParametrization::getHitXY(
    CLHEP::HepRandomEngine *engine, double &x, double &y) {
  x = CLHEP::RandGauss::shoot(engine, 0, m_sigmaX);
  y = CLHEP::RandGauss::shoot(engine, 0, m_sigmaY);
}
