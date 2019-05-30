/*
  Copyright (C) 2002-2017 CERN for the benefit of the ATLAS collaboration
*/

#include "FastCaloSimAnalyzer/TFCSValidationHitSpy.h"
#include "FastCaloSimAnalyzer/TFCSAnalyzerBase.h"

#include "ISF_FastCaloSimEvent/ICaloGeometry.h"
#include "ISF_FastCaloSimEvent/TFCSExtrapolationState.h"
#include <iostream>
// #include <tuple>
// #include <algorithm>
#include "TH1.h"
#include "TH2.h"

//=============================================
//======= TFCSValidationHitSpy =========
//=============================================

TFCSValidationHitSpy::TFCSValidationHitSpy(const char* name, const char* title, ICaloGeometry* geo) :
  TFCSLateralShapeParametrizationHitBase(name, title),
  m_geo(geo),
  m_previous(0),
  m_saved_cellele(0),
  m_hist_hitgeo_dphi(0),
  m_hist_hitgeo_matchprevious_dphi(0),
  m_hist_hitenergy_r(0),
  m_hist_hitenergy_z(0),
  m_hist_hitenergy_weight(0),
  m_hist_hitenergy_mean_r(0),
  m_hist_hitenergy_mean_z(0),
  m_hist_hitenergy_mean_weight(0),
  m_hist_Rz(0),
  m_hist_Rz_outOfRange(0),
  m_hist_hitenergy_alpha_radius(0),
  m_hist_hitenergy_alpha_absPhi_radius(0),
  m_hist_deltaEta(0),
  m_hist_deltaPhi(0),
  m_hist_deltaRt(0),
  m_hist_deltaZ(0),
  m_hist_total_hitPhi_minus_cellPhi(0),
  m_hist_matched_hitPhi_minus_cellPhi(0),
  m_hist_total_hitPhi_minus_cellPhi_etaboundary(0),
  m_hist_matched_hitPhi_minus_cellPhi_etaboundary(0)
{
}

FCSReturnCode TFCSValidationHitSpy::simulate_hit(Hit& hit, TFCSSimulationState& simulstate, const TFCSTruthState* truth, const TFCSExtrapolationState* extrapol)
{

  int cs = calosample();
  const int pdgId = truth->pdgid();
  double  charge   = getCharge(pdgId);
  
  const CaloDetDescrElement* cellele = 0;
  if (cs > 20) {
    float x = hit.x();
    float y = hit.y();
    float z = hit.z();
    cellele = m_geo->getFCalDDE(cs, x, y, z); // z is used for the detector side (A,C) determination only
    if (!cellele) {

      float r = sqrt(x * x + y * y + z * z);
      float eta = atanh(z / r);
      std::cout << "Error: Matching cell not found!" << " x: " << x << " y: " << y << " z: " << z << " eta: " << eta << std::endl;
      exit(-1);
      return FCSFatal;
    }
  }
  else cellele = m_geo->getDDE(cs, hit.eta(), hit.phi());

  if (1 == 1) {

    double dphi_hit = TFCSAnalyzerBase::DeltaPhi(hit.phi(), cellele->phi());
    
    float hitenergy = hit.E();
    float layer_energy = simulstate.E(cs);


    if (m_hist_hitgeo_dphi) m_hist_hitgeo_dphi->Fill(dphi_hit, hit.E());

    float extrapol_phi = hit.center_phi();
    float extrapol_r   = hit.center_r();
    float extrapol_z   = hit.center_z();
    float extrapol_eta = hit.center_eta();


    if (cs < 21) {

      float deta_hit_minus_extrapol = hit.eta() - extrapol_eta;
      float dphi_hit_minus_extrapol = TVector2::Phi_mpi_pi(hit.phi() - extrapol_phi);

      if(charge<0.)dphi_hit_minus_extrapol=-dphi_hit_minus_extrapol;
      if(extrapol_eta < 0.)deta_hit_minus_extrapol=-deta_hit_minus_extrapol;
      

      std::tie(m_deta_hit_minus_extrapol_mm, m_dphi_hit_minus_extrapol_mm) = TFCSAnalyzerBase::GetUnitsmm(extrapol_eta, deta_hit_minus_extrapol, dphi_hit_minus_extrapol, extrapol_r, extrapol_z);

      float alpha_mm = TMath::ATan2(m_dphi_hit_minus_extrapol_mm, m_deta_hit_minus_extrapol_mm);

      if (m_hist_deltaEta)m_hist_deltaEta->Fill(m_deta_hit_minus_extrapol_mm, hitenergy);
      if (m_hist_deltaPhi)m_hist_deltaPhi->Fill(m_dphi_hit_minus_extrapol_mm, hitenergy);
      if (m_hist_deltaRt)m_hist_deltaRt->Fill(hit.r() - extrapol_r, hitenergy);
      if (m_hist_deltaZ)m_hist_deltaZ->Fill(hit.z() - extrapol_z, hitenergy);

      if (m_hist_hitenergy_alpha_radius or m_hist_hitenergy_alpha_absPhi_radius) {

        float alpha_absPhi_mm = TMath::ATan2(TMath::Abs(m_dphi_hit_minus_extrapol_mm), m_deta_hit_minus_extrapol_mm);
        float radius_mm = TMath::Sqrt(m_dphi_hit_minus_extrapol_mm * m_dphi_hit_minus_extrapol_mm + m_deta_hit_minus_extrapol_mm * m_deta_hit_minus_extrapol_mm);

        if (alpha_mm < 0)
          alpha_mm = 2.0 * TMath::Pi() + alpha_mm;

        if (m_hist_hitenergy_alpha_radius and layer_energy > 0) {
          if (hitenergy < 0) hitenergy = 0;
          m_hist_hitenergy_alpha_radius->Fill(alpha_mm, radius_mm, hitenergy / layer_energy);
        }


        if (m_hist_hitenergy_alpha_absPhi_radius and layer_energy > 0 ) {
          if (hitenergy < 0) hitenergy = 0;
          m_hist_hitenergy_alpha_absPhi_radius->Fill(alpha_absPhi_mm, radius_mm, hitenergy / layer_energy);
        }

      }

    }
    else {
      /// Use x, y instread of eta, phi for FCal
      
      float hit_r=hit.r();
      float hit_phi=TMath::ATan2( hit.y(),hit.x() );
      float delta_r = hit_r - extrapol_r;
      float delta_phi_mm = TVector2::Phi_mpi_pi(hit_phi - extrapol_phi)*extrapol_r;
      if(charge<0.)delta_phi_mm=-delta_phi_mm;
      
      if (m_hist_hitenergy_alpha_radius or m_hist_hitenergy_alpha_absPhi_radius) {
	
	
	float radius_mm = sqrt(delta_r*delta_r + delta_phi_mm*delta_phi_mm) ;
        float alpha_mm = TMath::ATan2(delta_phi_mm, delta_r); 
        
	if (alpha_mm < 0)alpha_mm = 2.0 * TMath::Pi() + alpha_mm;

        

        if (m_hist_hitenergy_alpha_radius and layer_energy > 0) {
          if (hitenergy < 0) hitenergy = 0;
          m_hist_hitenergy_alpha_radius->Fill(alpha_mm, radius_mm, hitenergy / layer_energy);
        }

        if (m_hist_hitenergy_alpha_absPhi_radius and layer_energy > 0) {
          if (hitenergy < 0) hitenergy = 0;
          float alpha_absPhi_mm = TMath::ATan2(fabs(delta_phi_mm), delta_r);
          m_hist_hitenergy_alpha_absPhi_radius->Fill(alpha_absPhi_mm, radius_mm, hitenergy / layer_energy);
        }
      }
      

      if (m_hist_deltaEta)m_hist_deltaEta->Fill(delta_r, hitenergy);
      if (m_hist_deltaPhi)m_hist_deltaPhi->Fill(delta_phi_mm, hitenergy);
      if (m_hist_deltaRt)m_hist_deltaRt->Fill(hit.r() - extrapol_r, hitenergy);
      if (m_hist_deltaZ)m_hist_deltaZ->Fill(hit.z() - extrapol_z, hitenergy);
    }


    // for wiggle efficiency
    if ( m_hist_total_hitPhi_minus_cellPhi or m_hist_matched_hitPhi_minus_cellPhi) {


      bool is_consider_eta_boundary = false;
      bool is_matched = false;

      if (m_hist_matched_hitPhi_minus_cellPhi_etaboundary) is_consider_eta_boundary = true;

      if (previous()) {
        const CaloDetDescrElement* g4cellele  = previous()->saved_cellele();
        if (g4cellele->identify() == cellele->identify()) is_matched = true;
      }


      if (is_consider_eta_boundary) { // for layers where phi granularity changes at some eta
        float eta_boundary = get_eta_boundary();
        float cell_eta = cellele->eta();
        float cell_deta = cellele->deta();

        // do not consider the cells that lie across this eta boundary
        if ( TMath::Abs(cell_eta) < eta_boundary && (TMath::Abs(cell_eta) + 0.5 * cell_deta) < eta_boundary)
        {
          m_hist_total_hitPhi_minus_cellPhi->Fill(dphi_hit, hitenergy);
          if (is_matched) m_hist_matched_hitPhi_minus_cellPhi->Fill(dphi_hit, hitenergy);
        } else if ( TMath::Abs(cell_eta) > eta_boundary && (TMath::Abs(cell_eta) - 0.5 * cell_deta) > eta_boundary)
        {
          m_hist_total_hitPhi_minus_cellPhi_etaboundary->Fill(dphi_hit, hitenergy);
          if (is_matched) m_hist_matched_hitPhi_minus_cellPhi_etaboundary->Fill(dphi_hit, hitenergy);
        }
      } else { // for layers there is no change in phi granularity

        m_hist_total_hitPhi_minus_cellPhi->Fill(dphi_hit, hitenergy);
        if (is_matched) m_hist_matched_hitPhi_minus_cellPhi->Fill(dphi_hit, hitenergy);
      }
    }//end of wiggle efficiency

    if(m_hist_Rz){
      m_hist_Rz->Fill(hit.r(),hit.z(),hitenergy);
    }
    if (m_hist_hitenergy_weight) {
      float w(0.);
      if (m_geo->isCaloBarrel(cs) ){ // Barrel: weight from r
	w= (hit.r() - extrapol->r(cs, SUBPOS_ENT))/(extrapol->r(cs, SUBPOS_EXT) - extrapol->r(cs, SUBPOS_ENT) );
      }
      else{ // End-Cap and FCal: weight from z
	w = (hit.z() - extrapol->z(cs, SUBPOS_ENT))/(extrapol->z(cs, SUBPOS_EXT) - extrapol->z(cs, SUBPOS_ENT) );
      }
      if(m_hist_Rz_outOfRange && (w<0. || w>1.0) ) m_hist_Rz_outOfRange->Fill(hit.r(),hit.z());
      m_hist_hitenergy_weight->Fill(w,hitenergy);
      if( (cs!=3) && (w<=-0.25 || w >=1.25) ) ATH_MSG_DEBUG("Found weight outside [-0.25,1.25]: weight=" << w); // Weights are expected out of range in EMB3 (cs==3)
    }

    if (m_hist_hitenergy_r) {
      float r = hit.r();
      if (r > extrapol->r(cs, SUBPOS_ENT)) m_hist_hitenergy_r->Fill(r, hitenergy);
    }
    if (m_hist_hitenergy_z) {
      float z = fabs(hit.z());
      if (z > fabs(extrapol->z(cs, SUBPOS_ENT)))m_hist_hitenergy_z->Fill(z, hitenergy);
    }

    if (previous() && m_hist_hitgeo_matchprevious_dphi) {
      const CaloDetDescrElement* cellele_previous_hit = previous()->saved_cellele();
      if (cellele == cellele_previous_hit) m_hist_hitgeo_matchprevious_dphi->Fill(dphi_hit, hit.E());

    }

  }
  m_saved_hit = hit;
  m_saved_cellele = cellele;

  return FCSSuccess;
}

FCSReturnCode TFCSValidationHitSpy::simulate(TFCSSimulationState& /*simulstate*/, const TFCSTruthState* /*truth*/, const TFCSExtrapolationState* /*extrapol*/)
{

  if (m_hist_hitenergy_mean_r && m_hist_hitenergy_weight)  {
    double mean_weight = m_hist_hitenergy_weight->GetMean();
    if (mean_weight != 0.)m_hist_hitenergy_mean_weight->Fill(mean_weight);
    m_hist_hitenergy_mean_weight->GetXaxis()->SetTitle("<weight>");
    m_hist_hitenergy_weight->Reset();
  }

  if (m_hist_hitenergy_mean_r) {
    double mean_r = m_hist_hitenergy_r->GetMean();
    if (mean_r != 0.)m_hist_hitenergy_mean_r->Fill(mean_r);
    m_hist_hitenergy_mean_r->GetXaxis()->SetTitle("<R [mm]>");
    m_hist_hitenergy_r->Reset();
  }

  if (m_hist_hitenergy_mean_z) {
    double mean_z = m_hist_hitenergy_z->GetMean();
    if (mean_z != 0.)m_hist_hitenergy_mean_z->Fill(mean_z);
    m_hist_hitenergy_mean_z->GetXaxis()->SetTitle("<z [mm]>");
    m_hist_hitenergy_z->Reset();
  }

  return FCSSuccess;
}



void TFCSValidationHitSpy::Print(Option_t *option) const
{
  TString opt(option);
  bool shortprint = opt.Index("short") >= 0;
  bool longprint = msgLvl(MSG::DEBUG) || (msgLvl(MSG::INFO) && !shortprint);
  TString optprint = opt; optprint.ReplaceAll("short", "");

  TFCSLateralShapeParametrizationHitBase::Print(option);

  if (longprint) ATH_MSG_INFO(optprint << "  Previous hit spy=" << m_previous);
}


double TFCSValidationHitSpy::getCharge(const int pdgID){
  
  if(pdgID==11 || pdgID==211 || pdgID==2212) return 1.;
  else if (pdgID==-11 || pdgID==-211 || pdgID==-2212) return -1.;
  else if ( pdgID==22 || fabs(pdgID)==2112 || pdgID==111 ) return 0;
  else {
    ATH_MSG_FATAL(" This pdgID is not supported: " << pdgID);
  }
  
  return -999.;
}


//=============================================
//========== ROOT persistency stuff ===========
//=============================================

ClassImp(TFCSValidationHitSpy)
