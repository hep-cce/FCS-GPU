/*
  Copyright (C) 2002-2023 CERN for the benefit of the ATLAS collaboration
*/

#include "TROOT.h"

void runParamEtaSlice() {
    gROOT->ProcessLine(".x initTFCSAnalyzer.C");
    gROOT->ProcessLine(".x runTFCSCreateParamEtaSlice.cxx(@PID@, @EMIN@, @EMAX@, @ETAMIN@, @DIR@)");
}
