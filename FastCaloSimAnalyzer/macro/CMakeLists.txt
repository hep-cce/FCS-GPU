fcs_make_task(runTFCS2DParametrizationHistogram
  SOURCE runTFCS2DParametrizationHistogram.cxx
  DEPENDENCY ${FastCaloSimCommon_LIB}
  DEPENDENCY ${AthenaStandalone_LIB}
  DEPENDENCY ${EnergyParametrization_LIB}
  DEPENDENCY ${FastCaloSimAnalyzer_LIB}
)

fcs_make_task(runTFCSAverageShape
  SOURCE runTFCSAverageShape.cxx
  DEPENDENCY ${FastCaloSimCommon_LIB}
  DEPENDENCY ${AthenaStandalone_LIB}
  DEPENDENCY ${FastCaloSimAnalyzer_LIB}
)

fcs_make_task(runTFCSCreateParametrization
  SOURCE runTFCSCreateParametrization.cxx
  DEPENDENCY ${FastCaloSimCommon_LIB}
  DEPENDENCY ${AthenaStandalone_LIB}
  DEPENDENCY ${FastCaloSimAnalyzer_LIB}
)

fcs_make_task(runTFCSCreateParamEtaSlice
  SOURCE runTFCSCreateParamEtaSlice.cxx
  DEPENDENCY ${FastCaloSimCommon_LIB}
  DEPENDENCY ${AthenaStandalone_LIB}
  DEPENDENCY ${FastCaloSimAnalyzer_LIB}
)

fcs_make_task(runTFCSMergeParamEtaSlices
  SOURCE runTFCSMergeParamEtaSlices.cxx
  DEPENDENCY ${FastCaloSimCommon_LIB}
  DEPENDENCY ${AthenaStandalone_LIB}
  DEPENDENCY ${FastCaloSimAnalyzer_LIB}
)

fcs_make_task(runTFCSShapeValidation
  SOURCE runTFCSShapeValidation.cxx
  DEPENDENCY ${FastCaloSimCommon_LIB}
  DEPENDENCY ${AthenaStandalone_LIB}
  DEPENDENCY ${FastCaloSimAnalyzer_LIB}
)

fcs_make_task(runTFCSSimulation
  SOURCE runTFCSSimulation.cxx
  DEPENDENCY ${FastCaloSimCommon_LIB}
  DEPENDENCY ${AthenaStandalone_LIB}
  DEPENDENCY ${FastCaloSimAnalyzer_LIB}
)
fcs_make_task(runTFCSTests
  SOURCE runTFCSTests.cxx
  DEPENDENCY ${FastCaloSimCommon_LIB}
  DEPENDENCY ${AthenaStandalone_LIB}
  DEPENDENCY ${FastCaloSimAnalyzer_LIB}
)
