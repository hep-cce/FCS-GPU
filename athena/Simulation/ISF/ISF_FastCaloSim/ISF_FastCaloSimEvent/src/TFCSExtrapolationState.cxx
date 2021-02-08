/*
  Copyright (C) 2002-2017 CERN for the benefit of the ATLAS collaboration
*/

#include "ISF_FastCaloSimEvent/TFCSExtrapolationState.h"
#include <iostream>

//=============================================
//======= TFCSExtrapolationState =========
//=============================================

TFCSExtrapolationState::TFCSExtrapolationState()
{
  clear();
}

void TFCSExtrapolationState::Print(Option_t* ) const
{
  std::cout<<"IDCalo: eta="<<m_IDCaloBoundary_eta<<" phi="<<m_IDCaloBoundary_phi<<" r="<<m_IDCaloBoundary_r<<" z="<<m_IDCaloBoundary_z<<std::endl;
  for(int i=0;i<CaloCell_ID_FCS::MaxSample;++i) {
    if(m_CaloOK[i][SUBPOS_MID]) {
      std::cout<<"  layer "<<i<<" MID eta="<<m_etaCalo[i][SUBPOS_MID]<<" phi="<<m_phiCalo[i][SUBPOS_MID]<<" r="<<m_rCalo[i][SUBPOS_MID]<<" z="<<m_zCalo[i][SUBPOS_MID]<<std::endl;
    }
  }
}

void TFCSExtrapolationState::clear()
{
  for(int i=0;i<CaloCell_ID_FCS::MaxSample;++i) {
    for(int j=0;j<3;++j) {
      m_CaloOK[i][j]=false;
      m_etaCalo[i][j]=-999;
      m_phiCalo[i][j]=-999;
      m_rCalo[i][j]=0;
      m_zCalo[i][j]=0;
      m_dCalo[i][j]=0;
      m_distetaCaloBorder[i][j]=0;
    }
  }

  m_CaloSurface_sample = -1;
  m_CaloSurface_eta = -999;
  m_CaloSurface_phi = -999;
  m_CaloSurface_r = 0;
  m_CaloSurface_z = 0;

  m_IDCaloBoundary_eta = -999;
  m_IDCaloBoundary_phi = -999;
  m_IDCaloBoundary_r = 0;
  m_IDCaloBoundary_z = 0;

  m_IDCaloBoundary_AngleEta = -999;
  m_IDCaloBoundary_Angle3D = -999;
}
