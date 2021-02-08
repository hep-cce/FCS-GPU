/*
  Copyright (C) 2002-2018 CERN for the benefit of the ATLAS collaboration
*/

#ifndef ISF_FASTCALOSIMEVENT_TFCSParametrizationChain_h
#define ISF_FASTCALOSIMEVENT_TFCSParametrizationChain_h

#include "ISF_FastCaloSimEvent/TFCSParametrization.h"

class TFCSParametrizationChain:public TFCSParametrization {
public:
  TFCSParametrizationChain(const char* name=nullptr, const char* title=nullptr):TFCSParametrization(name,title) {};
  TFCSParametrizationChain(const TFCSParametrizationChain& ref):TFCSParametrization(ref.GetName(),ref.GetTitle()),m_chain(ref.chain()) {};

  ///Status bit for chain persistency
  enum FCSSplitChainObjects {
     kSplitChainObjects = BIT(16) ///< Set this bit in the TObject bit field if the TFCSParametrizationBase objects in the chain should be written as separate keys into the root file instead of directly writing the objects. This is needed if the sum of all objects in the chain use >1GB of memory, which can't be handeled by TBuffer. Drawback is that identical objects will get stored as multiple instances
  };

  bool SplitChainObjects() const {return TestBit(kSplitChainObjects);};
  void set_SplitChainObjects() {SetBit(kSplitChainObjects);};
  void reset_SplitChainObjects() {ResetBit(kSplitChainObjects);};

  typedef std::vector< TFCSParametrizationBase* > Chain_t;
  virtual unsigned int size() const override {return m_chain.size();};
  virtual const TFCSParametrizationBase* operator[](unsigned int ind) const override {return m_chain[ind];};
  virtual TFCSParametrizationBase* operator[](unsigned int ind) override {return m_chain[ind];};
  const Chain_t& chain() const {return m_chain;};
  Chain_t& chain() {return m_chain;};
  void push_back(const Chain_t::value_type& param) {m_chain.push_back(param);recalc();};

  virtual bool is_match_Ekin_bin(int Ekin_bin) const override;
  virtual bool is_match_calosample(int calosample) const override;

  virtual FCSReturnCode simulate(TFCSSimulationState& simulstate,const TFCSTruthState* truth, const TFCSExtrapolationState* extrapol) override;

  void Print(Option_t *option = "") const override;

  //THIS CLASS HAS A CUSTOM STREAMER! CHANGES IN THE VERSIONING OR DATA TYPES NEED TO BE IMPLEMENTED BY HAND!
  //void TFCSParametrizationChain::Streamer(TBuffer &R__b)
protected:
  void recalc_pdgid_intersect();
  void recalc_pdgid_union();

  void recalc_Ekin_intersect();
  void recalc_eta_intersect();
  void recalc_Ekin_eta_intersect();

  void recalc_Ekin_union();
  void recalc_eta_union();
  void recalc_Ekin_eta_union();
  
  ///Default is to call recalc_pdgid_intersect() and recalc_Ekin_eta_intersect()
  virtual void recalc();

  FCSReturnCode simulate_and_retry(TFCSParametrizationBase* parametrization, TFCSSimulationState& simulstate, const TFCSTruthState* truth, const TFCSExtrapolationState* extrapol);

private:  
  Chain_t m_chain;

  ClassDefOverride(TFCSParametrizationChain,2)  //TFCSParametrizationChain
};

#include "ISF_FastCaloSimEvent/TFCSParametrizationChain.icc"

#if defined(__ROOTCLING__) && defined(__FastCaloSimStandAlone__)
#pragma link C++ class TFCSParametrizationChain-;
#endif

#endif
