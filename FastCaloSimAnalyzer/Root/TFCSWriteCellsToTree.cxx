/*
  Copyright (C) 2002-2017 CERN for the benefit of the ATLAS collaboration
*/

#include "FastCaloSimAnalyzer/TFCSWriteCellsToTree.h"

#include "ISF_FastCaloSimEvent/TFCSSimulationState.h"
#include "CaloDetDescr/CaloDetDescrElement.h"

#include <iostream>

#include "TTree.h"

//=============================================
//======= TFCSWriteCellsToTree =========
//=============================================

TFCSWriteCellsToTree::TFCSWriteCellsToTree( const char* name, const char* title, TTree* tree )
    : TFCSParametrization( name, title ) {
  init_tree( tree );
}

void TFCSWriteCellsToTree::init_tree( TTree* tree ) {
  m_tree = tree;

  /** now add branches and leaves to the tree */
  if ( m_tree ) {
    m_oneeventcells = new FCS_matchedcellvector;
    m_tree->Branch( "AvgAllCells", &m_oneeventcells );

    // write cells per layer
    for ( Int_t i = 0; i < CaloCell_ID_FCS::MaxSample; i++ ) {
      TString branchname = "AvgSampling_";
      branchname += i;
      m_layercells[i] = new FCS_matchedcellvector;
      m_tree->Branch( branchname, &m_layercells[i] );
    }
  }
}

FCSReturnCode TFCSWriteCellsToTree::simulate( TFCSSimulationState& simulstate, const TFCSTruthState* /*truth*/,
                                              const TFCSExtrapolationState* /*extrapol*/ ) {
  if ( !m_tree ) return FCSFatal;

  m_oneeventcells->m_vector.clear();
  for ( Int_t i = 0; i < CaloCell_ID_FCS::MaxSample; i++ ) m_layercells[i]->m_vector.clear();

  // Now copy all cells into the tree
  for ( const auto& iter : simulstate.cells() ) {
    const CaloDetDescrElement* theDDE = iter.first;
    int                        layer  = theDDE->getSampling();

    FCS_matchedcell cell;
    cell.cell.cell_identifier = theDDE->identify();
    cell.cell.sampling        = layer;
    cell.cell.energy          = iter.second;
    cell.cell.center_x        = theDDE->x();
    cell.cell.center_y        = theDDE->y();
    cell.cell.center_z        = theDDE->z();

    m_oneeventcells->push_back( cell );
    m_layercells[layer]->push_back( cell );
  }

  m_tree->Fill();

  return FCSSuccess;
}

void TFCSWriteCellsToTree::Print( Option_t* option ) const {
  TString opt( option );
  bool    shortprint = opt.Index( "short" ) >= 0;
  bool    longprint  = msgLvl( MSG::DEBUG ) || ( msgLvl( MSG::INFO ) && !shortprint );
  TString optprint   = opt;
  optprint.ReplaceAll( "short", "" );

  TFCSParametrization::Print( option );

  if ( longprint ) ATH_MSG_INFO( optprint << "  tree ptr=" << m_tree );
}
