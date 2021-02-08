/*
  Copyright (C) 2002-2017 CERN for the benefit of the ATLAS collaboration
*/

#include "FastCaloSimAnalyzer/TFCSValidationEnergyAndCells.h"
#include "FastCaloSimAnalyzer/TFCSAnalyzerBase.h"

#include "ISF_FastCaloSimEvent/ICaloGeometry.h"
#include "ISF_FastCaloSimEvent/TFCSSimulationState.h"

#include <iostream>

//=============================================
//======= TFCSValidationEnergyAndCells =========
//=============================================

TFCSValidationEnergyAndCells::TFCSValidationEnergyAndCells( const char* name, const char* title,
                                                            TFCSAnalyzerBase* analysis )
    : TFCSLateralShapeParametrization( name, title ), m_analysis( analysis ) {}

FCSReturnCode TFCSValidationEnergyAndCells::simulate( TFCSSimulationState& simulstate, const TFCSTruthState* /*truth*/,
                                                      const TFCSExtrapolationState* /*extrapol*/ ) {
  if ( !analysis() ) return FCSFatal;
  simulstate.set_Ebin( analysis()->pca() );
  simulstate.set_E( analysis()->total_energy() );
  ATH_MSG_DEBUG( "Ebin=" << simulstate.Ebin() );
  ATH_MSG_DEBUG( "E=" << simulstate.E() );
  ATH_MSG_DEBUG( "E(" << calosample() << ")=" << simulstate.E( calosample() ) );
  for ( int i = 0; i < CaloCell_ID_FCS::MaxSample; ++i ) {
    simulstate.set_Efrac( i, analysis()->total_layer_cell_energy()[i] );
    simulstate.set_E( i, analysis()->total_layer_cell_energy()[i] * analysis()->total_energy() );
  }

  FCS_matchedcellvector* cellVector = analysis()->cellVector();
  if ( is_UseAvgShape() ) { cellVector = analysis()->avgcellVector(); }
  unsigned int ncells = cellVector->size();
  for ( unsigned int icell = 0; icell < ncells; icell++ ) {
    FCS_matchedcell&           matchedcell = cellVector->m_vector.at( icell );
    const CaloDetDescrElement* cellele     = m_geo->getDDE( matchedcell.cell.cell_identifier );
    simulstate.deposit( cellele, matchedcell.cell.energy );
  }
  return FCSSuccess;
}

void TFCSValidationEnergyAndCells::Print( Option_t* option ) const {
  TString opt( option );
  bool    shortprint = opt.Index( "short" ) >= 0;
  bool    longprint  = msgLvl( MSG::DEBUG ) || ( msgLvl( MSG::INFO ) && !shortprint );
  TString optprint   = opt;
  optprint.ReplaceAll( "short", "" );

  TFCSLateralShapeParametrization::Print( option );

  if ( longprint ) ATH_MSG_INFO( optprint << "  analysis ptr=" << m_analysis );
}
