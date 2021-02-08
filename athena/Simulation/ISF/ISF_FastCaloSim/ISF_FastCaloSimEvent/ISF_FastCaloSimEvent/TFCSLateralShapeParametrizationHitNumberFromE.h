/*
  Copyright (C) 2002-2018 CERN for the benefit of the ATLAS collaboration
*/

#ifndef TFCSLateralShapeParametrizationHitNumberFromE_h
#define TFCSLateralShapeParametrizationHitNumberFromE_h

#include "ISF_FastCaloSimEvent/TFCSLateralShapeParametrizationHitBase.h"

#include "TH2.h"


class TFCSLateralShapeParametrizationHitNumberFromE:public TFCSLateralShapeParametrizationHitBase {
public:
  /// LAr: 10.1%/sqrt(E)
  ///    stochastic=0.101;
  ///    constant=0.002;
  /// HadEC: 21.4%/sqrt(E)
  ///    stochastic=0.214;
  ///    constant=0.0;
  /// TileCal: 56.4%/sqrt(E)
  ///    stochastic=0.564;
  ///    constant=0.055;
  /// FCAL:    28.5%/sqrt(E)
  ///    stochastic=0.285;
  ///    constant=0.035;
  TFCSLateralShapeParametrizationHitNumberFromE(const char* name=nullptr, const char* title=nullptr,double stochastic=0.1,double constant=0);

  int get_number_of_hits(TFCSSimulationState& simulstate,const TFCSTruthState* truth, const TFCSExtrapolationState* extrapol) const override;

  void Print(Option_t *option = "") const override;
private:
  // simple shape information should be stored as private member variables here
  double m_stochastic;
  double m_constant;

  ClassDefOverride(TFCSLateralShapeParametrizationHitNumberFromE,1)  //TFCSLateralShapeParametrizationHitNumberFromE
};

#if defined(__ROOTCLING__) && defined(__FastCaloSimStandAlone__)
#pragma link C++ class TFCSLateralShapeParametrizationHitNumberFromE+;
#endif

#endif
