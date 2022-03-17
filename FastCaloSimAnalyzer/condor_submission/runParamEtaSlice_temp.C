#include "TROOT.h"

void runParamEtaSlice() {
    gROOT->ProcessLine(".x initTFCSAnalyzer.C");
    gROOT->ProcessLine(".x runTFCSCreateParamEtaSlice.cxx(@PID@, @EMIN@, @EMAX@, @ETAMIN@, @DIR@)");
}
