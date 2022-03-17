/*
  Copyright (C) 2002-2017 CERN for the benefit of the ATLAS collaboration
*/

#ifndef TFCSWriteCellsToTree_h
#define TFCSWriteCellsToTree_h

#include "ISF_FastCaloSimEvent/TFCSParametrization.h"
#include "ISF_FastCaloSimParametrization/FCS_Cell.h"
#include "ISF_FastCaloSimEvent/FastCaloSim_CaloCell_ID.h"

class TTree;

class TFCSWriteCellsToTree : public TFCSParametrization {
 public:
  TFCSWriteCellsToTree(const char* name = nullptr, const char* title = nullptr,
                       TTree* tree = nullptr);

  void init_tree(TTree* tree);

  virtual FCSReturnCode simulate(
      TFCSSimulationState& simulstate, const TFCSTruthState* truth,
      const TFCSExtrapolationState* extrapol) override;

  void Print(Option_t* option = "") const override;

 private:
  TTree* m_tree{nullptr};

  FCS_matchedcellvector* m_oneeventcells;  // these are all matched cells in a
                                           // single event
  FCS_matchedcellvector* m_layercells[CaloCell_ID_FCS::MaxSample];  // these are
                                                                    // all
                                                                    // matched
                                                                    // cells in
                                                                    // a given
                                                                    // layer in
                                                                    // a given
                                                                    // event

  ClassDefOverride(TFCSWriteCellsToTree, 1)  // TFCSWriteCellsToTree
};

#if defined(__MAKECINT__)
#pragma link C++ class TFCSWriteCellsToTree + ;
#endif

#endif
