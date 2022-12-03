/*
  Copyright (C) 2002-2023 CERN for the benefit of the ATLAS collaboration
*/

#define __FastCaloSimStandAloneDict__

#include "FastCaloSimAnalyzer/CaloGeometryFromFile.h"
#include "FastCaloSimAnalyzer/TFCS2DParametrization.h"
#include "FastCaloSimAnalyzer/TFCSAnalyzerBase.h"
#include "FastCaloSimAnalyzer/TFCSAnalyzerHelpers.h"
#include "FastCaloSimAnalyzer/TFCSEnergyInterpolation.h"
#include "FastCaloSimAnalyzer/TFCSFlatNtupleMaker.h"
#include "FastCaloSimAnalyzer/TFCSInputValidationPlots.h"
#include "FastCaloSimAnalyzer/TFCSHistoLateralShapeParametrizationFCal.h"
#include "FastCaloSimAnalyzer/TFCSHistoLateralShapeParametrization.h"
#include "FastCaloSimAnalyzer/TFCSHitCellMappingWiggle.h"
#include "FastCaloSimAnalyzer/TFCSLateralShapeParametrizationHitChain.h"
#include "FastCaloSimAnalyzer/TFCSShapeValidation.h"
#include "FastCaloSimAnalyzer/TFCSValidationEnergy.h"
#include "FastCaloSimAnalyzer/TFCSValidationEnergyAndCells.h"
#include "FastCaloSimAnalyzer/TFCSValidationEnergyAndHits.h"
#include "FastCaloSimAnalyzer/TFCSValidationHitSpy.h"
#include "FastCaloSimAnalyzer/TFCSVertexZPositionStudies.h"
#include "FastCaloSimAnalyzer/TFCSWriteCellsToTree.h"

#ifdef __ROOTCLING__

#pragma link C++ namespace FCS+;

#pragma link C++ class CaloGeometryFromFile+;
#pragma link C++ class TFCS2DParametrization+;
#pragma link C++ class TFCSAnalyzerBase+;
#pragma link C++ class TFCSEnergyInterpolation+;
#pragma link C++ class TFCSFlatNtupleMaker+;
#pragma link C++ class TFCSHistoLateralShapeParametrizationFCal+;
#pragma link C++ class TFCSHistoLateralShapeParametrization+;
#pragma link C++ class TFCSHitCellMappingWiggle+;
#pragma link C++ class TFCSInputValidationPlots+;
#pragma link C++ class TFCSLateralShapeParametrizationHitChain+;
#pragma link C++ class TFCSShapeValidation+;
#pragma link C++ class TFCSValidationEnergy+;
#pragma link C++ class TFCSValidationEnergyAndCells+;
#pragma link C++ class TFCSValidationEnergyAndHits+;
#pragma link C++ class TFCSValidationHitSpy+;
#pragma link C++ class TFCSVertexZPositionStudies+;
#pragma link C++ class TFCSWriteCellsToTree+;

#pragma link C++ struct FCS_truth+;

#endif
