/*
  Copyright (C) 2002-2018 CERN for the benefit of the ATLAS collaboration
*/

#ifndef TFCSHitCellMappingWiggle_h
#define TFCSHitCellMappingWiggle_h

#include "ISF_FastCaloSimEvent/TFCSHitCellMapping.h"

#ifdef USE_GPU
#  include "FastCaloGpu/FastCaloGpu/LoadGpuFuncHist.h"
#endif

class TFCS1DFunction;
class TH1;

class TFCSHitCellMappingWiggle : public TFCSHitCellMapping {

public:
  TFCSHitCellMappingWiggle( const char* name = nullptr, const char* title = nullptr, ICaloGeometry* geo = nullptr );
  ~TFCSHitCellMappingWiggle();

  void initialize( TFCS1DFunction* func );
  void initialize( const std::vector<const TFCS1DFunction*>& functions, const std::vector<float>& bin_low_edges );

  void initialize( TH1* histogram, float xscale = 1 );
  void initialize( const std::vector<const TH1*> histograms, std::vector<float> bin_low_edges, float xscale = 1 );

  inline unsigned int get_number_of_bins() const { return m_functions.size(); };

  inline double get_bin_low_edge( int bin ) const { return m_bin_low_edge[bin]; };
  inline double get_bin_up_edge( int bin ) const { return m_bin_low_edge[bin + 1]; };

  inline const TFCS1DFunction*             get_function( int bin ) { return m_functions[bin]; };
  const std::vector<const TFCS1DFunction*> get_functions() { return m_functions; };
  const std::vector<float>                 get_bin_low_edges() { return m_bin_low_edge; };

  /// modify one hit position to emulate the LAr accordeon shape
  /// and then fills all hits into calorimeter cells
  virtual FCSReturnCode simulate_hit( Hit& hit, TFCSSimulationState& simulstate, const TFCSTruthState* truth,
                                      const TFCSExtrapolationState* extrapol ) override;

  void Print( Option_t* option = "" ) const override;

  static void unit_test( TFCSSimulationState* simulstate = nullptr, TFCSTruthState* truth = nullptr,
                         TFCSExtrapolationState* extrapol = nullptr );
#ifdef USE_GPU
  void             set_d_HistFuncs( FHs* hf_ptr ) { m_d_HistFuncs = hf_ptr; };
  const FHs*       d_HistFuncs() { return m_d_HistFuncs; };
  void             LoadHistFuncs();
  LoadGpuFuncHist* LdFH() { return m_LdFH; };
#endif

private:
  //** Function for the hit-to-cell assignment accordion structure fix (wiggle)  **//
  //** To be moved to the conditions database at some point **//
  std::vector<const TFCS1DFunction*> m_functions    = {nullptr};
  std::vector<float>                 m_bin_low_edge = {0, static_cast<float>( init_eta_max )};

#ifdef USE_GPU
  FHs*             m_d_HistFuncs = nullptr;
  LoadGpuFuncHist* m_LdFH        = nullptr;
#endif
  ClassDefOverride( TFCSHitCellMappingWiggle, 1 ) // TFCSHitCellMappingWiggle
};

#if defined( __ROOTCLING__ ) && defined( __FastCaloSimStandAlone__ )
#  pragma link C++ class TFCSHitCellMappingWiggle + ;
#endif

#endif
