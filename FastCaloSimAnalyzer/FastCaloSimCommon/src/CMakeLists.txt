# Copyright (C) 2002-2018 CERN for the benefit of the ATLAS collaboration

# Sources
set(FastCaloSimCommon_Srcs
  # Dependencies
  ../dependencies/atlasrootstyle/AtlasLabels.C
  ../dependencies/atlasrootstyle/AtlasStyle.C
  ../dependencies/atlasrootstyle/AtlasUtils.C
  ../dependencies/docopt/docopt.cxx

  # CLHEP
  ../CLHEP/Random/RandFlat.cxx
  ../CLHEP/Random/RandGauss.cxx
  ../CLHEP/Random/RandomEngine.cxx
  ../CLHEP/Random/RandPoisson.cxx
  ../CLHEP/Random/TRandomEngine.cxx

  # HepPDT
  ../HepPDT/ParticleID.cxx

  # Dataset management
  SampleInfo.cxx
  TFCSSampleDiscovery.cxx
)

# XRootD support
if(ENABLE_XROOTD)
  list(APPEND FastCaloSimCommon_Srcs XrdStreamBuf.cxx)
endif()

# Global include is needed for dictionary generation to work
include_directories(
  .
  ..
  ../dependencies/
)

# ROOT dictionary
root_generate_dictionary(
  ${FastCaloSimCommon_LIB}Dict

  MODULE ${FastCaloSimCommon_LIB}
  LINKDEF LinkDef.h
)

# Define and build the library
fcs_add_library(${FastCaloSimCommon_LIB} SHARED ${FastCaloSimCommon_Srcs} ${FastCaloSimCommon_LIB}Dict.cxx)

target_link_libraries(${FastCaloSimCommon_LIB} ROOT::Core ROOT::Graf)

if(ENABLE_XROOTD)
  target_include_directories(${FastCaloSimCommon_LIB} PRIVATE ${XROOTD_INCLUDE_DIR})
  target_link_libraries(${FastCaloSimCommon_LIB} ${XROOTD_LIBRARIES})
endif()

# Let's mute AtlasStyle warnings
set_source_files_properties(../dependencies/atlasrootstyle/AtlasUtils.C PROPERTIES COMPILE_FLAGS -Wno-unused-variable)

# Install library
install(TARGETS ${FastCaloSimCommon_LIB}
  DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

# Install setup script
fcs_install_files(setup.sh)

# Install data
fcs_install_data(../data/DSID_DB.txt)
fcs_install_data(../data/InputSamplesList.txt)
