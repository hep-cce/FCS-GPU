/*
  Copyright (C) 2002-2018 CERN for the benefit of the ATLAS collaboration
*/

#ifndef ISF_FASTCALOSIMEVENT_TFCSParametrizationPDGIDSelectChain_h
#define ISF_FASTCALOSIMEVENT_TFCSParametrizationPDGIDSelectChain_h

#include "ISF_FastCaloSimEvent/TFCSParametrizationChain.h"

class TFCSParametrizationPDGIDSelectChain:public TFCSParametrizationChain {
public:
  TFCSParametrizationPDGIDSelectChain(const char* name=nullptr, const char* title=nullptr):TFCSParametrizationChain(name,title) {reset_SimulateOnlyOnePDGID();};
  TFCSParametrizationPDGIDSelectChain(const TFCSParametrizationPDGIDSelectChain& ref):TFCSParametrizationChain(ref) {reset_SimulateOnlyOnePDGID();};

  ///Status bit for PDGID Selection
  enum FCSPDGIDStatusBits {
     kSimulateOnlyOnePDGID = BIT(15) ///< Set this bit in the TObject bit field if the PDGID selection loop should be aborted after the first successful match
  };

  bool SimulateOnlyOnePDGID() const {return TestBit(kSimulateOnlyOnePDGID);};
  void set_SimulateOnlyOnePDGID() {SetBit(kSimulateOnlyOnePDGID);};
  void reset_SimulateOnlyOnePDGID() {ResetBit(kSimulateOnlyOnePDGID);};

  virtual FCSReturnCode simulate(TFCSSimulationState& simulstate,const TFCSTruthState* truth, const TFCSExtrapolationState* extrapol) override;

  static void unit_test(TFCSSimulationState* simulstate=nullptr,TFCSTruthState* truth=nullptr,TFCSExtrapolationState* extrapol=nullptr);
protected:
  virtual void recalc() override;

private:

  ClassDefOverride(TFCSParametrizationPDGIDSelectChain,1)  //TFCSParametrizationPDGIDSelectChain
};

#if defined(__ROOTCLING__) && defined(__FastCaloSimStandAlone__)
#pragma link C++ class TFCSParametrizationPDGIDSelectChain+;
#endif

#endif
