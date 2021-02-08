/*
  Copyright (C) 2002-2018 CERN for the benefit of the ATLAS collaboration
*/

#include "TFCSEnergyParametrizationPCABinCalculator.h"

#include "ISF_FastCaloSimEvent/TFCSSimulationState.h"
#include "ISF_FastCaloSimEvent/TFCSExtrapolationState.h"

//===================================================
//==== TFCSEnergyParametrizationPCABinCalculator ====
//===================================================

TFCSEnergyParametrizationPCABinCalculator::TFCSEnergyParametrizationPCABinCalculator( TFCSApplyFirstPCA applyfirstPCA,
                                                                                      const char*       name,
                                                                                      const char*       title )
    : TFCSEnergyParametrization( name, title ) {

  m_applyfirstPCA = applyfirstPCA;
  m_PCAbin        = -1;
}

FCSReturnCode TFCSEnergyParametrizationPCABinCalculator::simulate( TFCSSimulationState& simulstate,
                                                                   const TFCSTruthState* /*truth*/,
                                                                   const TFCSExtrapolationState* /*extrapol*/ ) {

  m_PCAbin = m_applyfirstPCA.get_PCAbin_from_simstate( simulstate );

  // cout<<"PCA bin "<<m_PCAbin<<endl;
  ATH_MSG( DEBUG ) << "PCA bin = " << m_PCAbin << std::endl;

  // simulstate.set_Ebin(m_PCAbin);

  return FCSSuccess;
}
