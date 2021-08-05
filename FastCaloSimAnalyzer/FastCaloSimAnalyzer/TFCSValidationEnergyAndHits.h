/*
  Copyright (C) 2002-2017 CERN for the benefit of the ATLAS collaboration
*/

#ifndef TFCSValidationEnergyAndHits_h
#define TFCSValidationEnergyAndHits_h

#include "TFCSLateralShapeParametrizationHitChain.h"
#include "TFCSValidationHitSpy.h"
#include <vector>
#include <tuple>
#include "TH2.h"

class TFCSAnalyzerBase;
class ICaloGeometry;
class TH1;
class TH2;

class TFCSValidationEnergyAndHits : public TFCSLateralShapeParametrizationHitChain {
public:
  TFCSValidationEnergyAndHits( const char* name = 0, const char* title = 0, TFCSAnalyzerBase* analysis = 0 );
  TFCSValidationEnergyAndHits( TFCSLateralShapeParametrizationHitBase* hitsim );

  virtual void   set_geometry( ICaloGeometry* geo ) override;
  ICaloGeometry* get_geometry() { return m_geo; };

  void                    set_analysis( TFCSAnalyzerBase* analysis ) { m_analysis = analysis; };
  TFCSAnalyzerBase*       analysis() { return m_analysis; };
  const TFCSAnalyzerBase* analysis() const { return m_analysis; };

  int n_bins() { return -1; }; // TO BE FIXED, SHOULD BE SOMEHOW READ FROM PCA FILE

  virtual FCSReturnCode simulate(TFCSSimulationState& simulstate,const TFCSTruthState* truth, const TFCSExtrapolationState* extrapol) override;

  virtual int get_number_of_hits(TFCSSimulationState& simulstate,const TFCSTruthState* truth, const TFCSExtrapolationState* extrapol) const override;

  void Print( Option_t* option = "" ) const override;

  TFCSValidationHitSpy& get_hitspy() { return m_hitspy; };

  void  add_histo( TH1* hist );
  TH1*& hist_hit_time() { return m_hist_hit_time; };
  TH2*& hist_problematic_hit_eta_phi() { return m_hist_problematic_hit_eta_phi; };
  TH1*& hist_deltaEtaAveragedPerEvent() { return m_hist_deltaEtaAveragedPerEvent; };
  TH1*& hist_energyPerLayer() { return m_hist_energyPerLayer; };

  std::vector<TH2*>   m_hist_ratioErecoEhit_vs_Ehit;
  std::vector<TH2*>   m_hist_ratioErecoEG4hit_vs_EG4hit;
  std::vector<double> m_hist_ratioErecoEhit_vs_Ehit_starttime;
  std::vector<double> m_hist_ratioErecoEhit_vs_Ehit_endtime;

  inline double                    getDeltaEtaAveraged() { return m_deltaEtaAveraged / m_energy_total; }
  inline std::pair<double,double> getMeanEnergyWithError(){return std::make_pair<double,double>( m_energy_total/m_eventCounter , sqrt( (m_energy_squared_total/m_eventCounter - pow(m_energy_total/m_eventCounter , 2) ) / m_eventCounter ) );}

private:
  ICaloGeometry*       m_geo;
  TFCSAnalyzerBase*    m_analysis;
  TFCSValidationHitSpy m_hitspy;
  std::vector<TH1*>    m_histos;
  TH1*                 m_hist_hit_time;
  TH2*                 m_hist_problematic_hit_eta_phi;

  TH1* m_hist_deltaEtaAveragedPerEvent;
  TH1* m_hist_energyPerLayer;

  double    m_deltaEtaAveraged;
  double    m_energy_total;
  double    m_energy_squared_total;
  double    m_deltaEtaAveragedPerEvent;
  long long m_eventCounter;

  ClassDefOverride( TFCSValidationEnergyAndHits, 1 ) // TFCSValidationEnergyAndHits
};

#if defined( __MAKECINT__ )
#  pragma link C++ class TFCSValidationEnergyAndHits + ;
#endif

#endif
