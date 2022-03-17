/*
  Copyright (C) 2002-2017 CERN for the benefit of the ATLAS collaboration
*/

#include "FastCaloSimAnalyzer/TFCSValidationEnergyAndHits.h"
#include "FastCaloSimAnalyzer/TFCSAnalyzerBase.h"

#include "ISF_FastCaloSimEvent/TFCSSimulationState.h"
#include "ISF_FastCaloSimEvent/ICaloGeometry.h"
#include "ISF_FastCaloSimEvent/TFCSLateralShapeParametrizationHitBase.h"
#include "ISF_FastCaloSimEvent/TFCSExtrapolationState.h"

#include <iostream>
#include "TH1.h"
#include "TH2.h"

//=============================================
//======= TFCSValidationEnergyAndHits =========
//=============================================

TFCSValidationEnergyAndHits::TFCSValidationEnergyAndHits(
    const char* name, const char* title, TFCSAnalyzerBase* analysis)
    : TFCSLateralShapeParametrizationHitChain(name, title),
      m_geo(nullptr),
      m_analysis(analysis),
      m_hist_hit_time(0),
      m_hist_problematic_hit_eta_phi(0),
      m_hist_deltaEtaAveragedPerEvent(0),
      m_hist_energyPerLayer(0),
      m_deltaEtaAveraged(0),
      m_energy_total(0.),
      m_energy_squared_total(0.),
      m_deltaEtaAveragedPerEvent(0.),
      m_eventCounter(0) {}

TFCSValidationEnergyAndHits::TFCSValidationEnergyAndHits(
    TFCSLateralShapeParametrizationHitBase* hitsim)
    : TFCSLateralShapeParametrizationHitChain(hitsim),
      m_geo(nullptr),
      m_analysis(nullptr),
      m_hist_hit_time(0),
      m_hist_problematic_hit_eta_phi(0),
      m_hist_deltaEtaAveragedPerEvent(0),
      m_hist_energyPerLayer(0),
      m_deltaEtaAveraged(0),
      m_energy_total(0.),
      m_energy_squared_total(0.),
      m_deltaEtaAveragedPerEvent(0.),
      m_eventCounter(0) {}

void TFCSValidationEnergyAndHits::set_geometry(ICaloGeometry* geo) {
  m_geo = geo;
  TFCSLateralShapeParametrizationHitChain::set_geometry(geo);
}

void TFCSValidationEnergyAndHits::add_histo(TH1* hist) {
  m_histos.push_back(hist);
}

int TFCSValidationEnergyAndHits::get_number_of_hits(
    TFCSSimulationState& /*simulstate*/, const TFCSTruthState* /*truth*/,
    const TFCSExtrapolationState* /*extrapol*/) const {
  int nhits = 0;
  unsigned int ncells = analysis()->cellVector()->size();
  for (unsigned int icell = 0; icell < ncells; icell++) {
    FCS_matchedcell& matchedcell = analysis()->cellVector()->m_vector.at(icell);
    nhits += matchedcell.hit.size();
  }

  return nhits;
}

FCSReturnCode TFCSValidationEnergyAndHits::simulate(
    TFCSSimulationState& simulstate, const TFCSTruthState* truth,
    const TFCSExtrapolationState* extrapol) {
  if (!analysis()) return FCSFatal;
  simulstate.set_Ebin(analysis()->pca());

  // cout << "Ebin: TFCSValidationEnergyAndHits: " << Ekin_bin() << "
  // TFCSSimulationState: " << simulstate.Ebin() << endl;

  if (Ekin_bin() >= 0 && Ekin_bin() != simulstate.Ebin()) return FCSFatal;

  ATH_MSG_DEBUG("Ebin: TFCSValidationEnergyAndHits: "
                << Ekin_bin() << " TFCSSimulationState: " << simulstate.Ebin());
  simulstate.set_E(analysis()->total_energy());
  for (int i = 0; i < CaloCell_ID_FCS::MaxSample; ++i) {
    simulstate.set_Efrac(i, analysis()->total_layer_cell_energy()[i]);
    simulstate.set_E(i, analysis()->total_layer_cell_energy()[i] *
                            analysis()->total_energy());
  }
  int cs = calosample();
  ATH_MSG_DEBUG("Ebin=" << simulstate.Ebin());
  ATH_MSG_DEBUG("E=" << simulstate.E());
  ATH_MSG_DEBUG("E(" << calosample() << ")=" << simulstate.E(calosample()));

  // m_hist_hit_dphi(0),m_hist_hitgeo_dphi(0),m_hist_hitgeo_wiggle_dphi(0)

  unsigned int ncells = analysis()->cellVector()->size();
  int tot_nhit = 0;

  m_deltaEtaAveragedPerEvent = 0.;
  double sum_energy_event = 0.;

  TFCSLateralShapeParametrizationHitBase::Hit hit;

  for (unsigned int icell = 0; icell < ncells; icell++) {
    FCS_matchedcell& matchedcell = analysis()->cellVector()->m_vector.at(icell);
    const CaloDetDescrElement* cellele =
        m_geo->getDDE(matchedcell.cell.cell_identifier);
    int nhit = matchedcell.hit.size();
    float sf = 1;
    float sum_hit_energy = 0;
    // if(nhit>0) sf=matchedcell.scalingfactor();
    std::vector<double> sum_hit_energy_in_time(
        m_hist_ratioErecoEhit_vs_Ehit.size(), 0);
    for (int i = 0; i < nhit; ++i) {
      const FCS_hit& h = matchedcell.hit[i];
      if (m_hist_hit_time) m_hist_hit_time->Fill(h.hit_time, h.hit_energy);

      TVector3 hitvec(h.hit_x, h.hit_y, h.hit_z);

      if (h.sampling > 20 && fabs(hitvec.Eta()) < 3.)
        std::cout << "Warning: Found hit in FCal layer which is not in FCal "
                     "eta range!\n"
                  << "Sampling: " << h.sampling
                  << " Cell identifier: " << std::hex << h.identifier
                  << std::dec << " energy: " << h.hit_energy
                  << " eta: " << hitvec.Eta() << " phi: " << hitvec.Phi()
                  << " z: " << h.hit_z << std::endl;

      if (h.sampling > 20)
        hit.setXYZE(h.hit_x, h.hit_y, h.hit_z, sf * h.hit_energy);
      else
        hit.setEtaPhiZE((float)hitvec.Eta(), (float)hitvec.Phi(), h.hit_z,
                        sf * h.hit_energy);

      m_hitspy.set_saved_hit(hit);
      m_hitspy.set_saved_cellele(cellele);
      for (TFCSLateralShapeParametrizationHitBase* hitsim : chain()) {
        if (msgLvl(MSG::DEBUG)) {
          if (tot_nhit < 2)
            hitsim->setLevel(MSG::DEBUG);
          else
            hitsim->setLevel(MSG::INFO);
        }
        hitsim->simulate_hit(hit, simulstate, truth, extrapol);
      }
      sum_hit_energy += h.hit_energy;
      for (unsigned int itime = 0; itime < sum_hit_energy_in_time.size();
           ++itime) {
        if (h.hit_time >= m_hist_ratioErecoEhit_vs_Ehit_starttime[itime] &&
            h.hit_time <= m_hist_ratioErecoEhit_vs_Ehit_endtime[itime]) {
          sum_hit_energy_in_time[itime] += h.hit_energy;
        }
      }
      m_deltaEtaAveragedPerEvent += (((TFCSValidationHitSpy*)(chain()[0]))
                                         ->get_deta_hit_minus_extrapol_mm()) *
                                    h.hit_energy;

      ++tot_nhit;
    }

    m_deltaEtaAveraged += m_deltaEtaAveragedPerEvent;
    sum_energy_event += sum_hit_energy;

    if (m_hist_energyPerLayer) m_hist_energyPerLayer->Fill(sum_hit_energy);

    float sum_G4hit_energy = 0;
    std::vector<double> sum_G4hit_energy_in_time(
        m_hist_ratioErecoEG4hit_vs_EG4hit.size(), 0);
    for (auto& g4hit : matchedcell.g4hit) {
      sum_G4hit_energy += g4hit.hit_energy;
      for (unsigned int itime = 0; itime < sum_G4hit_energy_in_time.size();
           ++itime) {
        if (g4hit.hit_time >= m_hist_ratioErecoEhit_vs_Ehit_starttime[itime] &&
            g4hit.hit_time <= m_hist_ratioErecoEhit_vs_Ehit_endtime[itime]) {
          sum_G4hit_energy_in_time[itime] += g4hit.hit_energy;
        }
      }
    }

    if (1 == 0) {
      std::cout << "cell cs=" << cs << " eta=" << cellele->eta()
                << " phi=" << cellele->phi() << " E=" << matchedcell.cell.energy
                << " EFCShit=" << sum_hit_energy
                << " EG4hit=" << sum_G4hit_energy << std::endl;
    }

    //    if(TMath::Abs(cellele->eta())>1.6 && TMath::Abs(cellele->eta())<2.0) {
    if (1 == 1) {
      for (auto hist : m_histos) {
        hist->Fill(sum_hit_energy, matchedcell.cell.energy);
      }
      double diff = TMath::Max(200.0, 0.05 * sum_hit_energy);
      if (TMath::Abs(sum_hit_energy - matchedcell.cell.energy) > diff) {
        // m_hist_problematic_hit_eta_phi->Fill(cellele->eta(),cellele->phi(),matchedcell.cell.energy-sum_hit_energy);
        if (m_hist_problematic_hit_eta_phi)
          m_hist_problematic_hit_eta_phi->Fill(cellele->eta(), cellele->phi(),
                                               1);
      }
      for (unsigned int itime = 0; itime < sum_hit_energy_in_time.size();
           ++itime) {
        m_hist_ratioErecoEhit_vs_Ehit[itime]
            ->Fill(sum_hit_energy_in_time[itime],
                   matchedcell.cell.energy / sum_hit_energy_in_time[itime]);
      }
      for (unsigned int itime = 0; itime < sum_G4hit_energy_in_time.size();
           ++itime) {
        m_hist_ratioErecoEG4hit_vs_EG4hit[itime]
            ->Fill(sum_G4hit_energy_in_time[itime],
                   matchedcell.cell.energy / sum_G4hit_energy_in_time[itime]);
      }
    }
  }

  m_deltaEtaAveragedPerEvent /= sum_energy_event;

  m_energy_total += (sum_energy_event / 1000.);
  m_energy_squared_total += pow(sum_energy_event / 1000., 2);
  m_eventCounter++;

  if (m_hist_deltaEtaAveragedPerEvent)
    m_hist_deltaEtaAveragedPerEvent->Fill(m_deltaEtaAveragedPerEvent);

  return FCSSuccess;
}

void TFCSValidationEnergyAndHits::Print(Option_t* option) const {
  TString opt(option);
  bool shortprint = opt.Index("short") >= 0;
  bool longprint = msgLvl(MSG::DEBUG) || (msgLvl(MSG::INFO) && !shortprint);
  TString optprint = opt;
  optprint.ReplaceAll("short", "");

  TFCSLateralShapeParametrizationHitChain::Print(option);

  if (longprint) ATH_MSG_INFO(optprint << "  analysis ptr=" << m_analysis);
}
