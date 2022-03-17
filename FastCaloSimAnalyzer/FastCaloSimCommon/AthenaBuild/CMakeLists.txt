# Copyright (C) 2002-2018 CERN for the benefit of the ATLAS collaboration

set(ATHENA_CALOGEO_PATH "${ATHENA_PATH}/Calorimeter/CaloGeoHelpers")
set(ATHENA_FCS_PATH "${ATHENA_PATH}/Simulation/ISF/ISF_FastCaloSim")

# Sources
set(AthenaStandalone_Srcs
  ${ATHENA_CALOGEO_PATH}/Root/CaloSampling.cxx

  ${ATHENA_FCS_PATH}/ISF_FastCaloSimEvent/src/IntArray.cxx
  ${ATHENA_FCS_PATH}/ISF_FastCaloSimEvent/src/TFCS1DFunction.cxx
  ${ATHENA_FCS_PATH}/ISF_FastCaloSimEvent/src/TFCS1DFunctionHistogram.cxx
  ${ATHENA_FCS_PATH}/ISF_FastCaloSimEvent/src/TFCS1DFunctionInt32Histogram.cxx
  ${ATHENA_FCS_PATH}/ISF_FastCaloSimEvent/src/TFCS1DFunctionRegression.cxx
  ${ATHENA_FCS_PATH}/ISF_FastCaloSimEvent/src/TFCS1DFunctionRegressionTF.cxx
  ${ATHENA_FCS_PATH}/ISF_FastCaloSimEvent/src/TFCSCenterPositionCalculation.cxx
  ${ATHENA_FCS_PATH}/ISF_FastCaloSimEvent/src/TFCSEnergyBinParametrization.cxx
  ${ATHENA_FCS_PATH}/ISF_FastCaloSimEvent/src/TFCSEnergyInterpolationLinear.cxx
  ${ATHENA_FCS_PATH}/ISF_FastCaloSimEvent/src/TFCSEnergyInterpolationSpline.cxx
  ${ATHENA_FCS_PATH}/ISF_FastCaloSimEvent/src/TFCSEnergyParametrization.cxx
  ${ATHENA_FCS_PATH}/ISF_FastCaloSimEvent/src/TFCSExtrapolationState.cxx
  ${ATHENA_FCS_PATH}/ISF_FastCaloSimEvent/src/TFCSHitCellMapping.cxx
  ${ATHENA_FCS_PATH}/ISF_FastCaloSimEvent/src/TFCSHitCellMappingFCal.cxx
  ${ATHENA_FCS_PATH}/ISF_FastCaloSimEvent/src/TFCSInvisibleParametrization.cxx
  ${ATHENA_FCS_PATH}/ISF_FastCaloSimEvent/src/TFCSLateralShapeParametrization.cxx
  ${ATHENA_FCS_PATH}/ISF_FastCaloSimEvent/src/TFCSLateralShapeParametrizationHitBase.cxx
  ${ATHENA_FCS_PATH}/ISF_FastCaloSimEvent/src/TFCSLateralShapeParametrizationHitNumberFromE.cxx
  ${ATHENA_FCS_PATH}/ISF_FastCaloSimEvent/src/TFCSParametrization.cxx
  ${ATHENA_FCS_PATH}/ISF_FastCaloSimEvent/src/TFCSParametrizationAbsEtaSelectChain.cxx
  ${ATHENA_FCS_PATH}/ISF_FastCaloSimEvent/src/TFCSParametrizationBase.cxx
  ${ATHENA_FCS_PATH}/ISF_FastCaloSimEvent/src/TFCSParametrizationBinnedChain.cxx
  ${ATHENA_FCS_PATH}/ISF_FastCaloSimEvent/src/TFCSParametrizationChain.cxx
  ${ATHENA_FCS_PATH}/ISF_FastCaloSimEvent/src/TFCSParametrizationEbinChain.cxx
  ${ATHENA_FCS_PATH}/ISF_FastCaloSimEvent/src/TFCSParametrizationEkinSelectChain.cxx
  ${ATHENA_FCS_PATH}/ISF_FastCaloSimEvent/src/TFCSParametrizationEtaSelectChain.cxx
  ${ATHENA_FCS_PATH}/ISF_FastCaloSimEvent/src/TFCSParametrizationFloatSelectChain.cxx
  ${ATHENA_FCS_PATH}/ISF_FastCaloSimEvent/src/TFCSParametrizationPDGIDSelectChain.cxx
  ${ATHENA_FCS_PATH}/ISF_FastCaloSimEvent/src/TFCSParametrizationPlaceholder.cxx
  ${ATHENA_FCS_PATH}/ISF_FastCaloSimEvent/src/TFCSPCAEnergyParametrization.cxx
  ${ATHENA_FCS_PATH}/ISF_FastCaloSimEvent/src/TFCSSimulationState.cxx
  ${ATHENA_FCS_PATH}/ISF_FastCaloSimEvent/src/TFCSTruthState.cxx
  ${ATHENA_FCS_PATH}/ISF_FastCaloSimEvent/src/TFCS2DFunctionHistogram.cxx
  ${ATHENA_FCS_PATH}/ISF_FastCaloSimEvent/src/TFCS2DFunction.cxx

  ${ATHENA_FCS_PATH}/ISF_FastCaloSimParametrization/src/CaloGeometry.cxx
  ${ATHENA_FCS_PATH}/ISF_FastCaloSimParametrization/src/CaloGeometryLookup.cxx
  ${ATHENA_FCS_PATH}/ISF_FastCaloSimParametrization/tools/CaloGeometryFromFile.cxx
  ${ATHENA_FCS_PATH}/ISF_FastCaloSimParametrization/tools/FCAL_ChannelMap.cxx
)

# Global include is needed for dictionary generation to work
include_directories(
  ${ATHENA_CALOGEO_PATH}
  ${ATHENA_FCS_PATH}
  ${ATHENA_FCS_PATH}/ISF_FastCaloSimEvent
  ${ATHENA_FCS_PATH}/ISF_FastCaloSimParametrization
  ${ATHENA_FCS_PATH}/ISF_FastCaloSimParametrization/tools
)

# ROOT dictionary
root_generate_dictionary(
  ${AthenaStandalone_LIB}Dict

  MODULE ${AthenaStandalone_LIB}
  LINKDEF LinkDef.h
)

# Define and build the library
fcs_add_library(${AthenaStandalone_LIB} SHARED ${AthenaStandalone_Srcs} ${AthenaStandalone_LIB}Dict.cxx)

fcs_add_dependency(${AthenaStandalone_LIB} ${FastCaloSimCommon_LIB})

target_link_libraries(${AthenaStandalone_LIB} ${FastCaloSimCommon_LIB} ROOT::Core ROOT::Gpad ROOT::Hist ROOT::Physics ROOT::Tree)

# Install library
install(TARGETS ${AthenaStandalone_LIB}
  DESTINATION ${CMAKE_INSTALL_LIBDIR}
)


# if(SYCL_TARGET_HIP)
#   find_package(hipSYCL CONFIG REQUIRED)
#   add_compile_options("-v -Wno-ignored-attributes --hipsycl-targets=hip:gfx900") # Silence HIP warnings
  # list(APPEND CMAKE_CXX_FLAGS "--hipsycl-targets=hip:gfx900")
  # add_definitions(-DSYCL_TARGET_HIP)
# endif()
