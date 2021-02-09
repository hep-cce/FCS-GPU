/*
  Copyright (C) 2002-2021 CERN for the benefit of the ATLAS collaboration
*/

#include "TROOT.h"

void runEpara() {
    gROOT->ProcessLine(".x initTFCSAnalyzer.C");
    gROOT->ProcessLine(".x run_epara.cxx(@DSID@, @METADATA@, @DIR@, @NPCA1@, @NPCA2@, @VALIDATION@, @VER@, @PLOTDIR@)");
}
