/*
  Copyright (C) 2002-2017 CERN for the benefit of the ATLAS collaboration
*/

#ifndef TFCSValidationHitSpy_h
#define TFCSValidationHitSpy_h

#include "ISF_FastCaloSimEvent/TFCSLateralShapeParametrizationHitBase.h"
#include "ISF_FastCaloSimEvent/TFCSSimulationState.h"
#include "ISF_FastCaloSimEvent/TFCSTruthState.h"
#include "TH2F.h"

class CaloGeometry;
class TH1;
class TH2;

class TFCSValidationHitSpy : public TFCSLateralShapeParametrizationHitBase {
public:
  TFCSValidationHitSpy( const char* name = 0, const char* title = 0, ICaloGeometry* geo = 0 );

  void           set_geometry( ICaloGeometry* geo ) override { m_geo = geo; };
  ICaloGeometry* get_geometry() { return m_geo; };

  // simulated one hit position with weight that should be put into simulstate
  // sometime later all hit weights should be resacled such that their final sum is simulstate->E(sample)
  // someone also needs to map all hits into cells
  virtual FCSReturnCode simulate_hit( Hit& hit, TFCSSimulationState& simulstate, const TFCSTruthState* truth,
                                      const TFCSExtrapolationState* extrapol ) override;

  virtual FCSReturnCode simulate( TFCSSimulationState& simulstate, const TFCSTruthState* truth,
                                  const TFCSExtrapolationState* extrapol ) override;

  void                        set_previous( TFCSValidationHitSpy* previous ) { m_previous = previous; };
  const TFCSValidationHitSpy* previous() const { return m_previous; };

  void       set_saved_hit( Hit& hit ) { m_saved_hit = hit; };
  const Hit& saved_hit() const { return m_saved_hit; };

  void                       set_saved_cellele( const CaloDetDescrElement* cellele ) { m_saved_cellele = cellele; };
  const CaloDetDescrElement* saved_cellele() const { return m_saved_cellele; };

  void  set_eta_boundary( float eta ) { m_phi_granularity_change_at_eta = eta; };
  float get_eta_boundary() { return m_phi_granularity_change_at_eta; };

  TH1*& hist_hitgeo_dphi() { return m_hist_hitgeo_dphi; };
  TH1*& hist_hitgeo_matchprevious_dphi() { return m_hist_hitgeo_matchprevious_dphi; };

  TH1*& hist_hitenergy_r() { return m_hist_hitenergy_r; };
  TH1*& hist_hitenergy_z() { return m_hist_hitenergy_z; };
  TH1*& hist_hitenergy_weight() { return m_hist_hitenergy_weight; };

  TH1*& hist_hitenergy_mean_r() { return m_hist_hitenergy_mean_r; };
  TH1*& hist_hitenergy_mean_z() { return m_hist_hitenergy_mean_z; };
  TH1*& hist_hitenergy_mean_weight() { return m_hist_hitenergy_mean_weight; };

  TH2*& hist_hitenergy_alpha_radius() { return m_hist_hitenergy_alpha_radius; };
  TH2*& hist_hitenergy_alpha_absPhi_radius() { return m_hist_hitenergy_alpha_absPhi_radius; };

  TH1*& hist_deltaEta() { return m_hist_deltaEta; };
  TH1*& hist_deltaPhi() { return m_hist_deltaPhi; };
  TH1*& hist_deltaRt() { return m_hist_deltaRt; };
  TH1*& hist_deltaZ() { return m_hist_deltaZ; };

  TH1*& hist_total_dphi() { return m_hist_total_hitPhi_minus_cellPhi; };
  TH1*& hist_matched_dphi() { return m_hist_matched_hitPhi_minus_cellPhi; };
  TH1*& hist_total_dphi_etaboundary() { return m_hist_total_hitPhi_minus_cellPhi_etaboundary; };
  TH1*& hist_matched_dphi_etaboundary() { return m_hist_matched_hitPhi_minus_cellPhi_etaboundary; };

  TH2*& hist_Rz() { return m_hist_Rz; };
  TH2*& hist_Rz_outOfRange() { return m_hist_Rz_outOfRange; };

  double getCharge( const int pdgID );

  double get_deta_hit_minus_extrapol_mm() { return m_deta_hit_minus_extrapol_mm; };
  double get_dphi_hit_minus_extrapol_mm() { return m_dphi_hit_minus_extrapol_mm; };

  void Print( Option_t* option ) const override;

private:
  // simple shape information should be stored as private member variables here

  ICaloGeometry*             m_geo; //! do not persistify
  TFCSValidationHitSpy*      m_previous;
  Hit                        m_saved_hit;
  const CaloDetDescrElement* m_saved_cellele;
  TH1*                       m_hist_hitgeo_dphi;
  TH1*                       m_hist_hitgeo_matchprevious_dphi;

  TH1* m_hist_hitenergy_r;
  TH1* m_hist_hitenergy_z;
  TH1* m_hist_hitenergy_weight;

  TH1* m_hist_hitenergy_mean_r;
  TH1* m_hist_hitenergy_mean_z;
  TH1* m_hist_hitenergy_mean_weight;

  TH2* m_hist_hitenergy_alpha_radius;
  TH2* m_hist_hitenergy_alpha_absPhi_radius;

  TH1* m_hist_deltaEta;
  TH1* m_hist_deltaPhi;
  TH1* m_hist_deltaRt;
  TH1* m_hist_deltaZ;

  TH2* m_hist_Rz;
  TH2* m_hist_Rz_outOfRange;

  TH1* m_hist_total_hitPhi_minus_cellPhi;
  TH1* m_hist_matched_hitPhi_minus_cellPhi;
  TH1* m_hist_total_hitPhi_minus_cellPhi_etaboundary;
  TH1* m_hist_matched_hitPhi_minus_cellPhi_etaboundary;

  double m_deta_hit_minus_extrapol_mm, m_dphi_hit_minus_extrapol_mm;
  float  m_phi_granularity_change_at_eta;

  ClassDefOverride( TFCSValidationHitSpy, 1 ) // TFCSValidationHitSpy
};

#if defined( __MAKECINT__ ) && defined( __FastCaloSimStandAlone__ )
#  pragma link C++ class TFCSValidationHitSpy + ;
#endif

#endif
