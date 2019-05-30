#define __FastCaloSimStandAloneDict__

#include "EnergyParametrizationValidation.h"
#include "firstPCA.h"
#include "secondPCA.h"
#include "TFCS1DFunctionFactory.h"
#include "TFCS1DRegression.h"
#include "TFCSApplyFirstPCA.h"
#include "TFCSEnergyParametrizationPCABinCalculator.h"
#include "TFCSMakeFirstPCA.h"
#include "TreeReader.h"


#ifdef __ROOTCLING__

#pragma link C++ class EnergyParametrizationValidation+;
#pragma link C++ class firstPCA+;
#pragma link C++ class secondPCA+;
#pragma link C++ class TFCS1DFunctionFactory+;
#pragma link C++ class TFCS1DRegression+;
#pragma link C++ class TFCSApplyFirstPCA+;
#pragma link C++ class TFCSEnergyParametrizationPCABinCalculator+;
#pragma link C++ class TFCSMakeFirstPCA+;
#pragma link C++ class TreeReader+;

#endif
