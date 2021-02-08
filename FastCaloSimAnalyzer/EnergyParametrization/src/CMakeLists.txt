# Copyright (C) 2002-2018 CERN for the benefit of the ATLAS collaboration

# Sources
set(EnergyParametrization_Srcs
  EnergyParametrizationValidation.cxx
  firstPCA.cxx
  secondPCA.cxx
  TFCS1DFunctionFactory.cxx
  TFCS1DRegression.cxx
  TFCSApplyFirstPCA.cxx
  TFCSEnergyParametrizationPCABinCalculator.cxx
  TFCSMakeFirstPCA.cxx
  TreeReader.cxx
)

# Headers
set(EnergyParametrization_Headers
  EnergyParametrizationValidation.h
  firstPCA.h
  secondPCA.h
  TFCS1DFunctionFactory.h
  TFCS1DRegression.h
  TFCSApplyFirstPCA.h
  TFCSEnergyParametrizationPCABinCalculator.h
  TFCSMakeFirstPCA.h
  TreeReader.h
)

# Add dictionary dependencies
fcs_dictionary_dependency(${AthenaStandalone_LIB})

# ROOT dictionary
root_generate_dictionary(
  ${EnergyParametrization_LIB}Dict

  MODULE ${EnergyParametrization_LIB}
  LINKDEF LinkDef.h
)

# Define and build the library
fcs_add_library(${EnergyParametrization_LIB} SHARED ${EnergyParametrization_Srcs} ${EnergyParametrization_LIB}Dict.cxx)

fcs_add_dependency(${EnergyParametrization_LIB} ${FastCaloSimCommon_LIB})
fcs_add_dependency(${EnergyParametrization_LIB} ${AthenaStandalone_LIB})

target_link_libraries(${EnergyParametrization_LIB} ROOT::Core ROOT::Graf ROOT::Tree ROOT::TMVA)

# Install library
install(TARGETS ${EnergyParametrization_LIB}
  DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

# Install headers
fcs_install_headers(${EnergyParametrization_Headers})
