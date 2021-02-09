/*
  Copyright (C) 2002-2021 CERN for the benefit of the ATLAS collaboration
*/

#include "TString.h"
#include <iostream>
#include <fstream>
#include <algorithm>
#include <string>
#include <stdlib.h>
#include <tuple>

void runWiggleAnalysis(){

  std::ifstream fin("WiggleInputs.txt");
  std::vector<int> DSIDs;

  while (fin){
    int DSID;
    fin >> DSID;
    DSIDs.push_back(DSID);
  }

  gROOT->ProcessLine(".x initTFCSAnalyzer.C");

  for(int i=0; i<DSIDs.size(); i++){
    TString command = ".x runTFCSWiggleDerivativeHistograms.C(" + std::to_string(DSIDs[i]) + ")";
    gROOT->ProcessLine(command);
  }

}
