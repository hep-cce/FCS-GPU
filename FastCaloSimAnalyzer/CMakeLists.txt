# Copyright (C) 2002-2018 CERN for the benefit of the ATLAS collaboration

cmake_minimum_required(VERSION 3.8 FATAL_ERROR)

# Add additional CMake macros and definitions
set(CMAKE_MODULE_PATH
  ${CMAKE_SOURCE_DIR}/FastCaloSimCommon/cmake
)

cmake_policy(SET CMP0074 NEW)

# Set project
project(FastCaloSimAnalyzer VERSION 1.0.0)

set(ENABLE_GPU OFF CACHE BOOL "Enable GPU")
set(RNDGEN_CPU OFF CACHE BOOL "Generate Random Numbers on CPU")
set(USE_STDPAR OFF CACHE BOOL "Use std::par")

if(USE_KOKKOS)
  find_package(Kokkos) 
endif()

include(FastCaloSim)
include(XRootD)



set(PROJECT_SRC_DIR ${CMAKE_SOURCE_DIR}/Root)

add_subdirectory(FastCaloSimCommon/src FastCaloSimCommon)
add_subdirectory(FastCaloSimCommon/AthenaBuild AthenaBuild)
add_subdirectory(EnergyParametrization/src EnergyParametrization)
add_subdirectory(Root)
if(ENABLE_GPU) 
  add_subdirectory(FastCaloGpu/src FastCaloGpu)
endif() 
add_subdirectory(macro)
