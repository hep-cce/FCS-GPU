/*
  Copyright (C) 2002-2018 CERN for the benefit of the ATLAS collaboration
*/

#include "ISF_FastCaloSimEvent/DoubleArray.h"
#include "ISF_FastCaloSimEvent/IntArray.h"
#include "ISF_FastCaloSimEvent/TFCSFunction.h"
#include "ISF_FastCaloSimEvent/TFCS1DFunction.h"
#include "ISF_FastCaloSimEvent/TFCS1DFunctionHistogram.h"
#include "ISF_FastCaloSimEvent/TFCS1DFunctionInt16Histogram.h"
#include "ISF_FastCaloSimEvent/TFCS1DFunctionInt32Histogram.h"
#include "ISF_FastCaloSimEvent/TFCS1DFunctionRegression.h"
#include "ISF_FastCaloSimEvent/TFCS1DFunctionRegressionTF.h"
#include "ISF_FastCaloSimEvent/TFCS1DFunctionSpline.h"
#include "ISF_FastCaloSimEvent/TFCS1DFunctionTemplateHistogram.h"
#include "ISF_FastCaloSimEvent/TFCS2DFunction.h"
#include "ISF_FastCaloSimEvent/TFCS2DFunctionHistogram.h"

#include "ISF_FastCaloSimEvent/TFCSParametrizationBase.h"
#include "ISF_FastCaloSimEvent/TFCSParametrizationPlaceholder.h"
#include "ISF_FastCaloSimEvent/TFCSParametrization.h"
#include "ISF_FastCaloSimEvent/TFCSInvisibleParametrization.h"
#include "ISF_FastCaloSimEvent/TFCSInitWithEkin.h"
#include "ISF_FastCaloSimEvent/TFCSEnergyInterpolationLinear.h"
#include "ISF_FastCaloSimEvent/TFCSEnergyInterpolationSpline.h"
#include "ISF_FastCaloSimEvent/TFCSParametrizationChain.h"
#include "ISF_FastCaloSimEvent/TFCSParametrizationBinnedChain.h"
#include "ISF_FastCaloSimEvent/TFCSParametrizationFloatSelectChain.h"
#include "ISF_FastCaloSimEvent/TFCSParametrizationPDGIDSelectChain.h"
#include "ISF_FastCaloSimEvent/TFCSParametrizationEbinChain.h"
#include "ISF_FastCaloSimEvent/TFCSParametrizationEkinSelectChain.h"
#include "ISF_FastCaloSimEvent/TFCSParametrizationEtaSelectChain.h"
#include "ISF_FastCaloSimEvent/TFCSParametrizationAbsEtaSelectChain.h"

#include "ISF_FastCaloSimEvent/TFCSEnergyParametrization.h"
#include "ISF_FastCaloSimEvent/TFCSPCAEnergyParametrization.h"
#include "ISF_FastCaloSimEvent/TFCSEnergyBinParametrization.h"

#include "ISF_FastCaloSimEvent/TFCSLateralShapeParametrization.h"
#include "ISF_FastCaloSimEvent/TFCSLateralShapeParametrizationHitBase.h"
#include "ISF_FastCaloSimEvent/TFCSLateralShapeParametrizationHitChain.h"
#include "ISF_FastCaloSimEvent/TFCSCenterPositionCalculation.h"
#include "ISF_FastCaloSimEvent/TFCSHistoLateralShapeParametrization.h"
#include "ISF_FastCaloSimEvent/TFCSHistoLateralShapeParametrizationFCal.h"
#include "ISF_FastCaloSimEvent/TFCSLateralShapeParametrizationHitNumberFromE.h"
#include "ISF_FastCaloSimEvent/TFCSHitCellMapping.h"
#include "ISF_FastCaloSimEvent/TFCSHitCellMappingFCal.h"
#include "ISF_FastCaloSimEvent/TFCSHitCellMappingWiggle.h"
#include "ISF_FastCaloSimEvent/TFCSHitCellMappingWiggleEMB.h"

#include "ISF_FastCaloSimEvent/TFCSTruthState.h"
#include "ISF_FastCaloSimEvent/TFCSExtrapolationState.h"
#include "ISF_FastCaloSimEvent/TFCSSimulationState.h"

#ifdef __CINT__
#pragma link C++ class DoubleArray+;
#pragma link C++ class IntArray+;
#pragma link C++ class TFCSFunction+;
#pragma link C++ class TFCS1DFunction+;
#pragma link C++ class TFCS1DFunctionHistogram+;
#pragma link C++ class TFCS1DFunctionInt16Histogram+;
#pragma link C++ class TFCS1DFunctionInt32Histogram+;
#pragma link C++ class TFCS1DFunctionRegression+;
#pragma link C++ class TFCS1DFunctionRegressionTF+;
#pragma link C++ class TFCS1DFunctionSpline+;

///Linkdefs needed for template based histograms
#pragma link C++ class TFCS1DFunction_Numeric<uint8_t,float>+;
#pragma link C++ class TFCS1DFunction_Numeric<uint16_t,float>+;
#pragma link C++ class TFCS1DFunction_Numeric<uint32_t,float>+;
#pragma link C++ class TFCS1DFunction_Numeric<float,float>+;
#pragma link C++ class TFCS1DFunction_Numeric<double,float>+;
#pragma link C++ class TFCS1DFunction_Numeric<double,double>+;

#pragma link C++ class TFCS1DFunction_Array<float>+;
#pragma link C++ class TFCS1DFunction_Array<double>+;
#pragma link C++ class TFCS1DFunction_Array<uint8_t>+;
#pragma link C++ class TFCS1DFunction_Array<uint16_t>+;
#pragma link C++ class TFCS1DFunction_Array<uint32_t>+;

#pragma link C++ class TFCS1DFunction_HistogramContent<float,float>+;
#pragma link C++ class TFCS1DFunction_HistogramContent<double,float>+;
#pragma link C++ class TFCS1DFunction_HistogramContent<double,double>+;
#pragma link C++ class TFCS1DFunction_HistogramContent<uint8_t,float>+;
#pragma link C++ class TFCS1DFunction_HistogramContent<uint16_t,float>+;
#pragma link C++ class TFCS1DFunction_HistogramContent<uint32_t,float>+;

#pragma link C++ class TFCS1DFunction_HistogramBinEdges<float,float>+;
#pragma link C++ class TFCS1DFunction_HistogramBinEdges<double,float>+;
#pragma link C++ class TFCS1DFunction_HistogramBinEdges<double,double>+;

#pragma link C++ class TFCS1DFunction_HistogramCompactBinEdges<float,uint8_t,float>+;
#pragma link C++ class TFCS1DFunction_HistogramCompactBinEdges<float,uint16_t,float>+;
#pragma link C++ class TFCS1DFunction_HistogramCompactBinEdges<float,uint32_t,float>+;

#pragma link C++ class TFCS1DFunction_HistogramInt8BinEdges+;
#pragma link C++ class TFCS1DFunction_HistogramInt16BinEdges+;
#pragma link C++ class TFCS1DFunction_HistogramInt32BinEdges+;
#pragma link C++ class TFCS1DFunction_HistogramFloatBinEdges+;
#pragma link C++ class TFCS1DFunction_HistogramDoubleBinEdges+;

#pragma link C++ class TFCS1DFunctionTemplateHistogram<TFCS1DFunction_HistogramInt8BinEdges,uint8_t,float>+;
#pragma link C++ class TFCS1DFunctionTemplateHistogram<TFCS1DFunction_HistogramInt8BinEdges,uint16_t,float>+;
#pragma link C++ class TFCS1DFunctionTemplateHistogram<TFCS1DFunction_HistogramInt8BinEdges,uint32_t,float>+;
#pragma link C++ class TFCS1DFunctionTemplateHistogram<TFCS1DFunction_HistogramInt16BinEdges,uint16_t,float>+;
#pragma link C++ class TFCS1DFunctionTemplateHistogram<TFCS1DFunction_HistogramInt16BinEdges,uint32_t,float>+;

#pragma link C++ class TFCS1DFunctionInt8Int8Histogram+;
#pragma link C++ class TFCS1DFunctionInt8Int16Histogram+;
#pragma link C++ class TFCS1DFunctionInt8Int32Histogram+;
#pragma link C++ class TFCS1DFunctionInt16Int16Histogram+;
#pragma link C++ class TFCS1DFunctionInt16Int32Histogram+;

#pragma link C++ class TFCS1DFunctionTemplateInterpolationHistogram<TFCS1DFunction_HistogramInt8BinEdges,uint8_t,float>+;
#pragma link C++ class TFCS1DFunctionTemplateInterpolationHistogram<TFCS1DFunction_HistogramInt8BinEdges,uint16_t,float>+;
#pragma link C++ class TFCS1DFunctionTemplateInterpolationHistogram<TFCS1DFunction_HistogramInt16BinEdges,uint16_t,float>+;
#pragma link C++ class TFCS1DFunctionTemplateInterpolationHistogram<TFCS1DFunction_HistogramInt16BinEdges,uint32_t,float>+;

#pragma link C++ class TFCS1DFunctionInt8Int8InterpolationHistogram+;
#pragma link C++ class TFCS1DFunctionInt8Int16InterpolationHistogram+;
#pragma link C++ class TFCS1DFunctionInt16Int16InterpolationHistogram+;
#pragma link C++ class TFCS1DFunctionInt16Int32InterpolationHistogram+;
///End Linkdefs needed for template based histograms

#pragma link C++ class TFCS2DFunction+;
#pragma link C++ class TFCS2DFunctionHistogram+;
#pragma link C++ class TFCSParametrizationBase+;
#pragma link C++ class TFCSParametrizationPlaceholder+;
#pragma link C++ class TFCSParametrization+;
#pragma link C++ class TFCSInvisibleParametrization+;
#pragma link C++ class TFCSInitWithEkin+;
#pragma link C++ class TFCSEnergyInterpolationLinear+;
#pragma link C++ class TFCSEnergyInterpolationSpline+;
#pragma link C++ class TFCSParametrizationChain-;
#pragma link C++ class TFCSParametrizationBinnedChain+;
#pragma link C++ class TFCSParametrizationFloatSelectChain+;
#pragma link C++ class TFCSParametrizationPDGIDSelectChain+;
#pragma link C++ class TFCSParametrizationEbinChain+;
#pragma link C++ class TFCSParametrizationEkinSelectChain+;
#pragma link C++ class TFCSParametrizationEtaSelectChain+;
#pragma link C++ class TFCSParametrizationAbsEtaSelectChain+;

#pragma link C++ class TFCSEnergyParametrization+;
#pragma link C++ class TFCSPCAEnergyParametrization+;
#pragma link C++ class TFCSEnergyBinParametrization+;

#pragma link C++ class TFCSLateralShapeParametrization+;
#pragma link C++ class TFCSLateralShapeParametrizationHitBase+;
#pragma link C++ class TFCSLateralShapeParametrizationHitChain+;
#pragma link C++ class TFCSCenterPositionCalculation+;
#pragma link C++ class TFCSHistoLateralShapeParametrization+;
#pragma link C++ class TFCSHistoLateralShapeParametrizationFCal+;
#pragma link C++ class TFCSLateralShapeParametrizationHitNumberFromE+;
#pragma link C++ class TFCSHitCellMapping+;
#pragma link C++ class TFCSHitCellMappingFCal+;
#pragma link C++ class TFCSHitCellMappingWiggle+;
#pragma link C++ class TFCSHitCellMappingWiggleEMB+;

#pragma link C++ class TFCSTruthState+;
#pragma link C++ class TFCSExtrapolationState+;
#pragma link C++ class TFCSSimulationState+;
#endif


