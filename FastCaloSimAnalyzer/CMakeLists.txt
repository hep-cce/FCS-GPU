# Copyright (C) 2002-2020 CERN for the benefit of the ATLAS collaboration

cmake_minimum_required(VERSION 3.8 FATAL_ERROR)

# Add additional CMake macros and definitions
set(CMAKE_MODULE_PATH
  ${CMAKE_SOURCE_DIR}/FastCaloSimCommon/cmake
)

# Set project
project(FastCaloSimAnalyzer VERSION 1.0.0)

set(ENABLE_GPU OFF CACHE BOOL "Enable GPU")
set(ENABLE_SYCL OFF CACHE BOOL "Enable SYCL/DPC++")

include(FastCaloSim)
include(XRootD)

set(PROJECT_SRC_DIR ${CMAKE_SOURCE_DIR}/Root)

add_subdirectory(FastCaloSimCommon/src FastCaloSimCommon)
add_subdirectory(FastCaloSimCommon/AthenaBuild AthenaBuild)
add_subdirectory(EnergyParametrization/src EnergyParametrization)
add_subdirectory(Root)
if(ENABLE_GPU)
  add_subdirectory(FastCaloGpu/src FastCaloGpu)
elseif(ENABLE_SYCL)
  add_subdirectory(FastCaloSycl FastCaloSycl)
endif() 
add_subdirectory(macro)
