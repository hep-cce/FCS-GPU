/*
  Copyright (C) 2002-2017 CERN for the benefit of the ATLAS collaboration
*/

#ifndef TFCSValidationEnergy_h
#define TFCSValidationEnergy_h

#include "ISF_FastCaloSimEvent/TFCSEnergyParametrization.h"

class TFCSAnalyzerBase;

class TFCSValidationEnergy:public TFCSEnergyParametrization {
public:
  TFCSValidationEnergy(const char* name=0, const char* title=0,TFCSAnalyzerBase* analysis=0);

  virtual bool is_match_calosample(int calosample) const override;
  virtual bool is_match_all_calosample() const override {return false;};
  virtual bool is_match_Ekin_bin(int Ekin_bin) const override;
  virtual bool is_match_all_Ekin_bin() const override {return true;};
  
  void set_analysis(TFCSAnalyzerBase* analysis) {m_analysis=analysis;};
  TFCSAnalyzerBase* analysis() {return m_analysis;};

  void set_n_bins(int n) {m_numberpcabins=n;};
  virtual int n_bins() const override {return m_numberpcabins;};
  std::vector<int>& get_layers() { return m_RelevantLayers; };
  const std::vector<int>& get_layers() const { return m_RelevantLayers; };

  virtual FCSReturnCode simulate(TFCSSimulationState& simulstate,const TFCSTruthState* truth, const TFCSExtrapolationState* extrapol) override;

  void Print(Option_t *option = "") const override;
private:
  int m_numberpcabins;
  std::vector<int>          m_RelevantLayers;

  TFCSAnalyzerBase* m_analysis;

  ClassDefOverride(TFCSValidationEnergy,1)  //TFCSValidationEnergy
};

#if defined(__MAKECINT__)
#pragma link C++ class TFCSValidationEnergy+;
#endif

#endif
