/*
  Copyright (C) 2002-2017 CERN for the benefit of the ATLAS collaboration
 */

#ifndef TFCSShapeValidation_H
#define TFCSShapeValidation_H

#include "CLHEP/Random/RandomEngine.h"
#include "TFCSAnalyzerBase.h"
#include "TFCSSimulationRun.h"
#include "CaloGeometryFromFile.h"
#include "ISF_FastCaloSimEvent/TFCSTruthState.h"
#include "ISF_FastCaloSimEvent/TFCSExtrapolationState.h"
#include "ISF_FastCaloSimEvent/TFCSSimulationState.h"

#if defined USE_GPU || defined USE_OMPGPU
#  include "FastCaloGpu/FastCaloGpu/GeoLoadGpu.h"
#endif
#include <chrono>

class TFCSShapeValidation : public TFCSAnalyzerBase {
public:
  TFCSShapeValidation( long seed = 42 );
  TFCSShapeValidation( TChain* chain, int layer, long seed = 42 );
  ~TFCSShapeValidation();

  void           LoadGeo();
  ICaloGeometry* get_geometry() { return m_geo; };

  void LoopEvents( int pcabin );

  TFCSTruthState&         get_truthTLV( int ievent ) { return m_truthTLV[ievent]; };
  TFCSExtrapolationState& get_extrapol( int ievent ) { return m_extrapol[ievent]; };

  std::vector<TFCSSimulationRun>& validations() { return m_validations; };
  int                             add_validation( TFCSParametrizationBase* sim ) {
    m_validations.emplace_back( sim );
    return m_validations.size() - 1;
  };
  int add_validation( const char* name, const char* title, TFCSParametrizationBase* sim ) {
    m_validations.emplace_back( name, title, sim );
    return m_validations.size() - 1;
  };

  void set_firstevent( int n ) { m_firstevent = n; };
  void set_nprint( int n ) { m_nprint = n; };

  int get_layer() const { return m_layer; };

  static std::chrono::duration<double> time_h;
  static std::chrono::duration<double> time_g1;
  static std::chrono::duration<double> time_g2;
  static std::chrono::duration<double> time_o1;
  static std::chrono::duration<double> time_o2;
  static std::chrono::duration<double> time_nhits;
  static std::chrono::duration<double> time_mchain;
  static std::chrono::duration<double> time_hitsim;
  static std::chrono::duration<double> time_reset;
  static std::chrono::duration<double> time_simA;
  static std::chrono::duration<double> time_reduce;
  static std::chrono::duration<double> time_copy;

#if defined USE_GPU || defined USE_OMPGPU
  void GeoLg();
  void region_data_cpy( CaloGeometryLookup* glkup, GeoRegion* gr );
  // void copy_all_regions( ) ;
#endif
private:
  CLHEP::HepRandomEngine* m_randEngine;

  TChain*               m_chain;
  std::string           m_output;
  int                   m_layer;
  int                   m_nprint;
  int                   m_firstevent;
  CaloGeometryFromFile* m_geo;

  std::vector<TFCSTruthState>         m_truthTLV;
  std::vector<TFCSExtrapolationState> m_extrapol;

  std::vector<TFCSSimulationRun> m_validations;

#if defined USE_GPU || defined USE_OMPGPU
  GeoLoadGpu* m_gl;
  void*       m_rd4h;
#endif

  ClassDef( TFCSShapeValidation, 1 );
};

#if defined( __MAKECINT__ )
#  pragma link C++ class TFCSShapeValidation + ;
#endif

#endif
