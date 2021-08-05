/*
  Copyright (C) 2002-2017 CERN for the benefit of the ATLAS collaboration
*/

#ifndef TFCSValidationEnergyAndCells_h
#define TFCSValidationEnergyAndCells_h

#include "ISF_FastCaloSimEvent/TFCSLateralShapeParametrization.h"

class TFCSAnalyzerBase;
class CaloGeometry;

class TFCSValidationEnergyAndCells : public TFCSLateralShapeParametrization {
public:
  TFCSValidationEnergyAndCells( const char* name = 0, const char* title = 0, TFCSAnalyzerBase* analysis = 0 );

  /// Status bit for FCS needs
  enum FCSStatusBits {
     kUseAvgShape = BIT(15) ///< Set this bit in the TObject bit field if the cell energies for the avg shape should be used
  };

  virtual bool is_UseAvgShape() const { return TestBit( kUseAvgShape ); };
  virtual void set_UseAvgShape() { SetBit( kUseAvgShape ); };
  virtual void reset_UseAvgShape() { ResetBit( kUseAvgShape ); };

  virtual void   set_geometry( ICaloGeometry* geo ) override { m_geo = geo; };
  ICaloGeometry* get_geometry() { return m_geo; };

  void              set_analysis( TFCSAnalyzerBase* analysis ) { m_analysis = analysis; };
  TFCSAnalyzerBase* analysis() { return m_analysis; };

  int n_bins() { return -1; }; // TO BE FIXED, SHOULD BE SOMEHOW READ FROM PCA FILE

  virtual FCSReturnCode simulate(TFCSSimulationState& simulstate,const TFCSTruthState* truth, const TFCSExtrapolationState* extrapol) override;

  void Print( Option_t* option = "" ) const override;

private:
  ICaloGeometry* m_geo;

  TFCSAnalyzerBase* m_analysis;

  ClassDefOverride( TFCSValidationEnergyAndCells, 1 ) // TFCSValidationEnergyAndCells
};

#if defined( __MAKECINT__ )
#  pragma link C++ class TFCSValidationEnergyAndCells + ;
#endif

#endif
