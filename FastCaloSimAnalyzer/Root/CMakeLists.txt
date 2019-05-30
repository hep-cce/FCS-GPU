# Copyright (C) 2002-2018 CERN for the benefit of the ATLAS collaboration

# Sources
set(FastCaloSimAnalyzer_Srcs
  CaloGeometryFromFile.cxx
  TFCS2DParametrization.cxx
  TFCSAnalyzerBase.cxx
  TFCSAnalyzerHelpers.cxx
  TFCSEnergyInterpolation.cxx
  TFCSFlatNtupleMaker.cxx
  TFCSInputValidationPlots.cxx
  TFCSShapeValidation.cxx
  TFCSValidationEnergy.cxx
  TFCSValidationEnergyAndCells.cxx
  TFCSValidationEnergyAndHits.cxx
  TFCSValidationHitSpy.cxx
  TFCSVertexZPositionStudies.cxx
  TFCSWriteCellsToTree.cxx
)

# # Global include is needed for dictionary generation to work
include_directories(../)

# Add dictionary dependencies
fcs_dictionary_dependency(${AthenaStandalone_LIB})

# ROOT dictionary
root_generate_dictionary(
  ${FastCaloSimAnalyzer_LIB}Dict

  MODULE ${FastCaloSimAnalyzer_LIB}
  LINKDEF LinkDef.h
)

# Define and build the library
message(STATUS "Creating library target '${FastCaloSimAnalyzer_LIB}'")
add_library(${FastCaloSimAnalyzer_LIB} SHARED ${FastCaloSimAnalyzer_Srcs} ${FastCaloSimAnalyzer_LIB}Dict.cxx)
target_compile_definitions(${FastCaloSimAnalyzer_LIB} PRIVATE ${FCS_CommonDefinitions})
target_link_libraries(${FastCaloSimAnalyzer_LIB} ${AthenaStandalone_LIB} ROOT::Core ROOT::Graf ROOT::Tree ROOT::TMVA)

if(ENABLE_XROOTD)
  target_include_directories(${FastCaloSimAnalyzer_LIB} PRIVATE ${XROOTD_HOME}/include/xrootd)
  target_link_libraries(${FastCaloSimAnalyzer_LIB} ${XROOTD_HOME}/lib64/libXrdCl.so)
endif()

fcs_add_dependency(${FastCaloSimAnalyzer_LIB} ${FastCaloSimCommon_LIB})
fcs_add_dependency(${FastCaloSimAnalyzer_LIB} ${AthenaStandalone_LIB})

# Install library
install(TARGETS ${FastCaloSimAnalyzer_LIB}
  DESTINATION ${CMAKE_INSTALL_LIBDIR}
)
