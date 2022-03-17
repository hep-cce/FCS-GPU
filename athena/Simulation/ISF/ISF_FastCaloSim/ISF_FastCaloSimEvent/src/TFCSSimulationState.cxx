/*
  Copyright (C) 2002-2017 CERN for the benefit of the ATLAS collaboration
*/

#include "CLHEP/Random/RandomEngine.h"

#include "ISF_FastCaloSimEvent/TFCSSimulationState.h"
#include <iostream>

//=============================================
//======= TFCSSimulationState =========
//=============================================

TFCSSimulationState::TFCSSimulationState(CLHEP::HepRandomEngine* randomEngine)
    : m_randomEngine(randomEngine) {
  clear();
}

void TFCSSimulationState::clear() {
  m_SF = 1;
  m_Ebin = -1;
  m_Etot = 0;
  for (int i = 0; i < CaloCell_ID_FCS::MaxSample; ++i) {
    m_E[i] = 0;
    m_Efrac[i] = 0;
  }
}

void TFCSSimulationState::deposit(const CaloDetDescrElement* cellele, float E) {
  // std::cout<<"TFCSSimulationState::deposit: cellele="<<cellele<<" E="<<E;
  auto mele = m_cells.find(cellele);
  if (mele == m_cells.end()) {
    m_cells.emplace(cellele, 0);
    mele = m_cells.find(cellele);
  }
  // std::cout<<" Ebefore="<<mele->second;
  m_cells[cellele] += E;
  // std::cout<<" Eafter="<<mele->second;
  // std::cout<<std::endl;
}

void TFCSSimulationState::Print(Option_t*) const {
  std::cout << "Ebin=" << m_Ebin << " E=" << E() << " #cells=" << m_cells.size()
            << std::endl;
  for (int i = 0; i < CaloCell_ID_FCS::MaxSample; ++i)
    if (E(i) != 0) {
      std::cout << "  E" << i << "(" << CaloSampling::getSamplingName(i)
                << ")=" << E(i) << " E" << i << "/E=" << Efrac(i) << std::endl;
    }
}
