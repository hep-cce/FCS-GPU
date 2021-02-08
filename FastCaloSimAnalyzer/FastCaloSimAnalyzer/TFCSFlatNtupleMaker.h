/*
  Copyright (C) 2002-2017 CERN for the benefit of the ATLAS collaboration
 */

#ifndef TFCSFlatNtupleMaker_H
#define TFCSFlatNtupleMaker_H

#include "TFCSAnalyzerBase.h"

class TFCSFlatNtupleMaker : public TFCSAnalyzerBase {
public:
  TFCSFlatNtupleMaker();
  TFCSFlatNtupleMaker( TChain*, TString, std::vector<int> );
  ~TFCSFlatNtupleMaker();

  void InitHistos( int event );
  void GetShowerCenter();
  void LoopEvents();
  void StudyHitMerging();
  void BookFlatNtuple( TTree* );

private:
  int m_debug;

  TChain*          m_chain;
  std::string      m_output;
  std::vector<int> m_vlayer;

  int      b_m_ievent;
  bool     b_m_new_event;
  bool     b_m_new_cell;
  Long64_t b_m_cell_identifier;
  Long64_t b_m_identifier;
  int      b_m_layer;
  int      b_m_pca;
  float    b_m_truth_eta;
  float    b_m_truth_phi;
  float    b_m_truth_energy;
  float    b_m_TTC_eta;
  float    b_m_TTC_phi;
  float    b_m_TTC_r;
  float    b_m_TTC_z;
  float    b_m_deta;
  float    b_m_dphi;
  float    b_m_deta_mm;
  float    b_m_dphi_mm;
  float    b_m_energy;
  float    b_m_scalefactor;
  float    b_m_hit_time;
  float    b_m_cell_energy;
  float    b_m_r;
  float    b_m_alpha;
  float    b_m_r_mm;
  float    b_m_alpha_mm;
  float    b_m_dx;
  float    b_m_dy;

  ClassDef( TFCSFlatNtupleMaker, 1 );
};

#if defined( __MAKECINT__ )
#  pragma link C++ class TFCSFlatNtupleMaker + ;
#endif

#endif
