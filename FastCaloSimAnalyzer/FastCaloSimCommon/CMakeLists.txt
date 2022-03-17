# Copyright (C) 2002-2018 CERN for the benefit of the ATLAS collaboration

cmake_minimum_required(VERSION 3.8 FATAL_ERROR)

# Add additional CMake macros and definitions
set(CMAKE_MODULE_PATH
  ${CMAKE_SOURCE_DIR}/cmake
)

# Set project
project(FastCaloSimCommon VERSION 1.0.0)

include(FastCaloSim)

# FastCaloSimCommon build
add_subdirectory(src)

# Standalone Athena FCS build
add_subdirectory(AthenaBuild)

if(SYCL_TARGET_HIP)
  find_package(hipSYCL CONFIG REQUIRED)
#   add_compile_options("-Wno-ignored-attributes") # Silence HIP warnings
#   list(APPEND CMAKE_CXX_FLAGS "--hipsycl-targets=hip:gfx900")
#   add_definitions(-DSYCL_TARGET_HIP)
endif()
