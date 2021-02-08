/*
  Copyright (C) 2002-2018 CERN for the benefit of the ATLAS collaboration
*/

#ifndef TFCSSimpleLateralShapeParametrization_h
#define TFCSSimpleLateralShapeParametrization_h

#include "ISF_FastCaloSimEvent/TFCSLateralShapeParametrizationHitBase.h"

#include "TFile.h"
#include "TH2F.h"
#include "TF1.h"

namespace CLHEP {
  class HepRandomEngine;
}

class TFCSSimpleLateralShapeParametrization:public TFCSLateralShapeParametrizationHitBase {
public:
  TFCSSimpleLateralShapeParametrization(const char* name=nullptr, const char* title=nullptr);

  // simulated one hit position with weight that should be put into simulstate
  // sometime later all hit weights should be resacled such that their final sum is simulstate->E(sample)
  // someone also needs to map all hits into cells
  virtual FCSReturnCode simulate_hit(Hit& hit,TFCSSimulationState& simulstate,const TFCSTruthState* truth, const TFCSExtrapolationState* extrapol) override;

  // Init and fill sigma
  bool Initialize(const char* filepath, const char* histname);

  bool Initialize(float input_sigma_x, float input_sigma_y);

  void getHitXY(CLHEP::HepRandomEngine *engine, double &x, double &y);

  float getSigma_x(){return m_sigmaX;};
  float getSigma_y(){return m_sigmaY;};
private:
  // simple shape information should be stored as private member variables here

  float m_sigmaX;
  float m_sigmaY;

  ClassDefOverride(TFCSSimpleLateralShapeParametrization,1)  //TFCSSimpleLateralShapeParametrization
};

#if defined(__ROOTCLING__) && defined(__FastCaloSimStandAlone__)
#pragma link C++ class TFCSSimpleLateralShapeParametrization+;
#endif

#endif
