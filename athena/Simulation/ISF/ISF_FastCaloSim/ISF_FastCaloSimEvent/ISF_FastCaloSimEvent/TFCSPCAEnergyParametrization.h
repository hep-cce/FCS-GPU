/*
  Copyright (C) 2002-2018 CERN for the benefit of the ATLAS collaboration
*/

#ifndef ISF_FASTCALOSIMEVENT_TFCSPCAEnergyParametrization_h
#define ISF_FASTCALOSIMEVENT_TFCSPCAEnergyParametrization_h


#include "ISF_FastCaloSimEvent/TFCSEnergyParametrization.h"
#include "ISF_FastCaloSimEvent/TFCS1DFunction.h"
#include "ISF_FastCaloSimEvent/IntArray.h"
#include "TMatrixF.h"
#include "TMatrixDSym.h"
#include "TVectorF.h"
#include "TFile.h"

class TFCSPCAEnergyParametrization:public TFCSEnergyParametrization
{
 public:
  TFCSPCAEnergyParametrization(const char* name=nullptr, const char* title=nullptr);

  virtual FCSReturnCode simulate(TFCSSimulationState& simulstate,const TFCSTruthState* truth, const TFCSExtrapolationState* extrapol) override;
  
  int n_pcabins() const { return m_numberpcabins; };
  virtual int n_bins() const override {return m_numberpcabins;};
  const std::vector<int>& get_layers() const { return m_RelevantLayers; };

  virtual bool is_match_Ekin_bin(int Ekin_bin) const override;
  virtual bool is_match_calosample(int calosample) const override;
  virtual bool is_match_all_Ekin_bin() const override {return true;};
  virtual bool is_match_all_calosample() const override {return false;};
  
  void P2X(TVectorD*, TVectorD* , TMatrixD* , int, double* , double* , int);
  bool loadInputs(TFile* file);
  bool loadInputs(TFile* file,std::string);
  
  void clean();
  
  void Print(Option_t *option = "") const override;

  int                       do_rescale;
  
 private:
  
  std::vector<int>          m_RelevantLayers;

  std::vector<TMatrixD*>    m_EV;
  std::vector<TVectorD*>    m_MeanValues;
  std::vector<TVectorD*>    m_SigmaValues;
  std::vector<TVectorD*>    m_Gauss_means;
  std::vector<TVectorD*>    m_Gauss_rms;
  std::vector<std::vector<TFCS1DFunction*> > m_cumulative;
  
  int m_numberpcabins;
  
  ClassDefOverride(TFCSPCAEnergyParametrization,1)  //TFCSPCAEnergyParametrization
 
};

#if defined(__ROOTCLING__) && defined(__FastCaloSimStandAlone__)
#pragma link C++ class TFCSPCAEnergyParametrization+;
#endif

#endif
