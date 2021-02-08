#define __FastCaloSimStandAloneDict__

#include "ISF_FastCaloSimEvent/IntArray.h"
#include "ISF_FastCaloSimEvent/TFCS1DFunction.h"
#include "ISF_FastCaloSimEvent/TFCS1DFunctionHistogram.h"
#include "ISF_FastCaloSimEvent/TFCS1DFunctionInt32Histogram.h"
#include "ISF_FastCaloSimEvent/TFCS1DFunctionRegression.h"
#include "ISF_FastCaloSimEvent/TFCS1DFunctionRegressionTF.h"
#include "ISF_FastCaloSimEvent/TFCS2DFunction.h"
#include "ISF_FastCaloSimEvent/TFCS2DFunctionHistogram.h"
#include "ISF_FastCaloSimEvent/TFCSCenterPositionCalculation.h"
#include "ISF_FastCaloSimEvent/TFCSEnergyBinParametrization.h"
#include "ISF_FastCaloSimEvent/TFCSEnergyInterpolationLinear.h"
#include "ISF_FastCaloSimEvent/TFCSEnergyInterpolationSpline.h"
#include "ISF_FastCaloSimEvent/TFCSEnergyParametrization.h"
#include "ISF_FastCaloSimEvent/TFCSExtrapolationState.h"
#include "ISF_FastCaloSimEvent/TFCSFunction.h"
#include "ISF_FastCaloSimEvent/TFCSHitCellMapping.h"
#include "ISF_FastCaloSimEvent/TFCSHitCellMappingFCal.h"
#include "ISF_FastCaloSimEvent/TFCSInvisibleParametrization.h"
#include "ISF_FastCaloSimEvent/TFCSLateralShapeParametrization.h"
#include "ISF_FastCaloSimEvent/TFCSLateralShapeParametrizationHitBase.h"
#include "ISF_FastCaloSimEvent/TFCSLateralShapeParametrizationHitNumberFromE.h"
#include "ISF_FastCaloSimEvent/TFCSPCAEnergyParametrization.h"
#include "ISF_FastCaloSimEvent/TFCSParametrization.h"
#include "ISF_FastCaloSimEvent/TFCSParametrizationAbsEtaSelectChain.h"
#include "ISF_FastCaloSimEvent/TFCSParametrizationBase.h"
#include "ISF_FastCaloSimEvent/TFCSParametrizationBinnedChain.h"
#include "ISF_FastCaloSimEvent/TFCSParametrizationChain.h"
#include "ISF_FastCaloSimEvent/TFCSParametrizationEbinChain.h"
#include "ISF_FastCaloSimEvent/TFCSParametrizationEkinSelectChain.h"
#include "ISF_FastCaloSimEvent/TFCSParametrizationEtaSelectChain.h"
#include "ISF_FastCaloSimEvent/TFCSParametrizationFloatSelectChain.h"
#include "ISF_FastCaloSimEvent/TFCSParametrizationPDGIDSelectChain.h"
#include "ISF_FastCaloSimEvent/TFCSParametrizationPlaceholder.h"
#include "ISF_FastCaloSimEvent/TFCSSimulationState.h"
#include "ISF_FastCaloSimEvent/TFCSTruthState.h"

#include "ISF_FastCaloSimParametrization/CaloGeometry.h"
#include "ISF_FastCaloSimParametrization/CaloGeometryLookup.h"
#include "ISF_FastCaloSimParametrization/FCS_Cell.h"

#ifdef __ROOTCLING__

#  pragma link C++ class IntArray + ;
#  pragma link C++ class TFCSFunction + ;
#  pragma link C++ class TFCS1DFunction + ;
#  pragma link C++ class TFCS1DFunctionHistogram + ;
#  pragma link C++ class TFCS1DFunctionInt32Histogram + ;
#  pragma link C++ class TFCS1DFunctionRegression + ;
#  pragma link C++ class TFCS1DFunctionRegressionTF + ;
#  pragma link C++ class TFCSCenterPositionCalculation + ;
#  pragma link C++ class TFCSEnergyBinParametrization + ;
#  pragma link C++ class TFCSEnergyInterpolationLinear + ;
#  pragma link C++ class TFCSEnergyInterpolationSpline + ;
#  pragma link C++ class TFCSEnergyParametrization + ;
#  pragma link C++ class TFCSExtrapolationState + ;
#  pragma link C++ class TFCSHitCellMapping + ;
#  pragma link C++ class TFCSHitCellMappingFCal + ;
#  pragma link C++ class TFCSHitCellMappingWiggle + ;
#  pragma link C++ class TFCSInvisibleParametrization + ;
#  pragma link C++ class TFCSLateralShapeParametrization + ;
#  pragma link C++ class TFCSLateralShapeParametrizationHitBase + ;
#  pragma link C++ class TFCSLateralShapeParametrizationHitNumberFromE + ;
#  pragma link C++ class TFCSParametrization + ;
#  pragma link C++ class TFCSParametrizationAbsEtaSelectChain + ;
#  pragma link C++ class TFCSParametrizationBase + ;
#  pragma link C++ class TFCSParametrizationBinnedChain + ;
#  pragma link C++ class TFCSParametrizationChain - ;
#  pragma link C++ class TFCSParametrizationEbinChain + ;
#  pragma link C++ class TFCSParametrizationEkinSelectChain + ;
#  pragma link C++ class TFCSParametrizationEtaSelectChain + ;
#  pragma link C++ class TFCSParametrizationFloatSelectChain + ;
#  pragma link C++ class TFCSParametrizationPDGIDSelectChain + ;
#  pragma link C++ class TFCSParametrizationPlaceholder + ;
#  pragma link C++ class TFCSPCAEnergyParametrization + ;
#  pragma link C++ class TFCSSimulationState + ;
#  pragma link C++ class TFCSTruthState + ;
#  pragma link C++ class TFCS2DFunctionHistogram + ;
#  pragma link C++ class TFCS2DFunction + ;

#  pragma link C++ class CaloGeometry + ;
#  pragma link C++ class CaloGeometryLookup + ;
#  pragma link C++ struct FCS_cell + ;
#  pragma link C++ struct FCS_g4hit + ;
#  pragma link C++ struct FCS_hit + ;
#  pragma link C++ struct FCS_matchedcell + ;
#  pragma link C++ struct FCS_matchedcellvector + ;

#endif
