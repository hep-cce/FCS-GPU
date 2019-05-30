/*
  Copyright (C) 2002-2018 CERN for the benefit of the ATLAS collaboration
*/

#ifndef TFCSHistoLateralShapeParametrizationFCal_h
#define TFCSHistoLateralShapeParametrizationFCal_h

#include "ISF_FastCaloSimEvent/TFCSHistoLateralShapeParametrization.h"
#include "ISF_FastCaloSimEvent/TFCS2DFunctionHistogram.h"
#include "ISF_FastCaloSimEvent/TFCSTruthState.h"

class TH2;

class TFCSHistoLateralShapeParametrizationFCal:public TFCSHistoLateralShapeParametrization {
public:
  TFCSHistoLateralShapeParametrizationFCal(const char* name=nullptr, const char* title=nullptr);
  ~TFCSHistoLateralShapeParametrizationFCal();

  virtual FCSReturnCode simulate_hit(Hit& hit,TFCSSimulationState& simulstate,const TFCSTruthState* truth, const TFCSExtrapolationState* extrapol) override;
  
private:
 
  ClassDefOverride(TFCSHistoLateralShapeParametrizationFCal,1)  //TFCSHistoLateralShapeParametrizationFCal
};

#if defined(__ROOTCLING__) && defined(__FastCaloSimStandAlone__)
#pragma link C++ class TFCSHistoLateralShapeParametrizationFCal+;
#endif

#endif
