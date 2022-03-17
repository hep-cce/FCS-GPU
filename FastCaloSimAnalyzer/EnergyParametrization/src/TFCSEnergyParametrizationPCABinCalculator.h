#ifndef TFCSEnergyParametrizationPCABinCalculator_h
#define TFCSEnergyParametrizationPCABinCalculator_h

//#include "ISF_FastCaloSimEvent/TFCSParametrization.h"
#include "ISF_FastCaloSimEvent/TFCSEnergyParametrization.h"
#include "ISF_FastCaloSimEvent/TFCSSimulationState.h"
#include "TFCSApplyFirstPCA.h"

class TFCSEnergyParametrizationPCABinCalculator
    : public TFCSEnergyParametrization {

 public:
  TFCSEnergyParametrizationPCABinCalculator(TFCSApplyFirstPCA applyfirstPCA,
                                            const char* name = nullptr,
                                            const char* title = nullptr);

  virtual FCSReturnCode simulate(
      TFCSSimulationState& simulstate, const TFCSTruthState* truth,
      const TFCSExtrapolationState* extrapol) override;

  int PCAbin() { return m_PCAbin; }

 private:
  TFCSApplyFirstPCA m_applyfirstPCA;

  int m_PCAbin;

  ClassDefOverride(TFCSEnergyParametrizationPCABinCalculator,
                   1)  // TFCSEnergyParametrizationPCABinCalculator
};

#if defined(__ROOTCLING__) && defined(__FastCaloSimStandAlone__)
#pragma link C++ class TFCSEnergyParametrizationPCABinCalculator + ;
#endif

#endif
