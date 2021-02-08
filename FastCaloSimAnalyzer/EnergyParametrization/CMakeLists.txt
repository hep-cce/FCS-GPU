# Copyright (C) 2002-2018 CERN for the benefit of the ATLAS collaboration

cmake_minimum_required(VERSION 3.8 FATAL_ERROR)

# Add additional CMake macros and definitions
set(CMAKE_MODULE_PATH
  ${CMAKE_SOURCE_DIR}/FastCaloSimCommon/cmake
)

# Set project
project(EnergyParametrization VERSION 1.0.0)

include(FastCaloSim)

set(PROJECT_SRC_DIR ${CMAKE_SOURCE_DIR}/src)

add_subdirectory(FastCaloSimCommon/src FastCaloSimCommon)
add_subdirectory(FastCaloSimCommon/AthenaBuild AthenaBuild)

add_subdirectory(src)
add_subdirectory(scripts)
