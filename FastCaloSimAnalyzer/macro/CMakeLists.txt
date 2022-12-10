if (ENABLE_SYCL)
  # set( CMAKE_C_COMPILER "clang" )
  # set( CMAKE_CXX_COMPILER "clang++" )
  # set( CMAKE_CXX_FLAGS "-g -O2 -fsycl -Wno-unknown-cuda-version" )
  set( CMAKE_CXX_FLAGS "-fsycl" )
  
  fcs_make_task(runTFCS2DParametrizationHistogram
    SOURCE runTFCS2DParametrizationHistogram.cxx
    DEPENDENCY ${FastCaloSimCommon_LIB}
    DEPENDENCY ${AthenaStandalone_LIB}
    DEPENDENCY ${EnergyParametrization_LIB}
    DEPENDENCY ${FastCaloSimAnalyzer_LIB}
    DEPENDENCY ${FastCaloSycl_LIB}
  )
  
  fcs_make_task(runTFCSAverageShape
    SOURCE runTFCSAverageShape.cxx
    DEPENDENCY ${FastCaloSimCommon_LIB}
    DEPENDENCY ${AthenaStandalone_LIB}
    DEPENDENCY ${FastCaloSimAnalyzer_LIB}
    DEPENDENCY ${FastCaloSycl_LIB}
  )
  
  fcs_make_task(runTFCSCreateParametrization
    SOURCE runTFCSCreateParametrization.cxx
    DEPENDENCY ${FastCaloSimCommon_LIB}
    DEPENDENCY ${AthenaStandalone_LIB}
    DEPENDENCY ${FastCaloSimAnalyzer_LIB}
    DEPENDENCY ${FastCaloSycl_LIB}
  )
  
  fcs_make_task(runTFCSCreateParamEtaSlice
    SOURCE runTFCSCreateParamEtaSlice.cxx
    DEPENDENCY ${FastCaloSimCommon_LIB}
    DEPENDENCY ${AthenaStandalone_LIB}
    DEPENDENCY ${FastCaloSimAnalyzer_LIB}
    DEPENDENCY ${FastCaloSycl_LIB}
  )
  
  fcs_make_task(runTFCSMergeParamEtaSlices
    SOURCE runTFCSMergeParamEtaSlices.cxx
    DEPENDENCY ${FastCaloSimCommon_LIB}
    DEPENDENCY ${AthenaStandalone_LIB}
    DEPENDENCY ${FastCaloSimAnalyzer_LIB}
    DEPENDENCY ${FastCaloSycl_LIB}
  )
  
  fcs_make_task(runTFCSShapeValidation
    SOURCE runTFCSShapeValidation.cxx
    DEPENDENCY ${FastCaloSimCommon_LIB}
    DEPENDENCY ${AthenaStandalone_LIB}
    DEPENDENCY ${FastCaloSimAnalyzer_LIB}
    DEPENDENCY ${FastCaloSycl_LIB}
  )
  
  fcs_make_task(runTFCSSimulation
    SOURCE runTFCSSimulation.cxx
    DEPENDENCY ${FastCaloSimCommon_LIB}
    DEPENDENCY ${AthenaStandalone_LIB}
    DEPENDENCY ${FastCaloSimAnalyzer_LIB}
    DEPENDENCY ${FastCaloSycl_LIB}
  )
else()
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
endif()
