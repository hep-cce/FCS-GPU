# Copyright (C) 2002-2018 CERN for the benefit of the ATLAS collaboration

cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

# Add additional CMake macros and definitions
set(CMAKE_MODULE_PATH
  ${CMAKE_SOURCE_DIR}/FastCaloSimCommon/cmake
)

cmake_policy(SET CMP0074 NEW)

# Set project
project(FastCaloSimAnalyzer VERSION 1.0.0)

set(ENABLE_GPU OFF CACHE BOOL "Enable GPU")
set(ENABLE_OMPGPU OFF CACHE BOOL "Enable OMP GPU")
set(RNDGEN_CPU OFF CACHE BOOL "Generate Random Numbers on CPU")
set(USE_STDPAR OFF CACHE BOOL "Use std::par")
set(STDPAR_TARGET "gpu" CACHE STRING "target for std::par. should be gpu,multicore or cpu")
set(DUMP_HITCELLS OFF CACHE BOOL "printout hit cell contents")
set(USE_ATOMIC_ADD OFF CACHE BOOL "use atomicAdd on float and int structs")
set(USE_ALPAKA OFF CACHE BOOL "Use alpaka")
set(USE_HIP OFF CACHE BOOL "Use HIP")
set(HIP_TARGET "AMD" CACHE STRING "HIP backend. must be either AMD or NVIDIA")

if ( USE_STDPAR )
  SET(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
  if ( ${STDPAR_TARGET} STREQUAL "cpu" )
    if ( NOT RNDGEN_CPU )
      message(WARNING "when STDPAR_TARGET=cpu, RNDGEN_CPU must be ON")
      message(WARNING "Setting RNDGEN_CPU to ON")
      set( RNDGEN_CPU ON )
    endif()
    set(STDPAR_DIRECTIVE "-nostdpar")
  elseif( ${STDPAR_TARGET} STREQUAL "gpu" )
    set(STDPAR_DIRECTIVE "-stdpar=gpu")
  elseif( ${STDPAR_TARGET} STREQUAL "multicore" )
    if ( USE_ATOMIC_ADD )
      message(WARNING "when STDPAR_TARGET=multicore, USE_ATOMIC_ADD must be OFF")
      message(WARNING "Setting USE_ATOMIC_ADD to OFF")
      set ( USE_ATOMIC_ADD OFF )
    endif()
    if ( NOT RNDGEN_CPU )
      message(WARNING "when STDPAR_TARGET=multicore, RNDGEN_CPU must be ON")
      message(WARNING "Setting RNDGEN_CPU to ON")
      set( RNDGEN_CPU ON )
    endif()
    set(STDPAR_DIRECTIVE "-stdpar=multicore")
  else()
    message(FATAL_ERROR "unknown stdpar target ${STDPAR_TARGET}")
  endif()
  message (STATUS "Will target ${STDPAR_TARGET} for std::par with ${STDPAR_DIRECTIVE}")

elseif(USE_KOKKOS)
  find_package(Kokkos) 

elseif(USE_ALPAKA)
  find_package(alpaka REQUIRED)

elseif(USE_HIP)
  if ( ${HIP_TARGET} STREQUAL "NVIDIA" )
    find_package(HIP)  
    if ( NOT RNDGEN_CPU )
      message(FATAL_ERROR "when HIP_TARGET=NVIDIA, RNDGEN_CPU must be ON")
    endif()
  else()
    find_package(HIP REQUIRED)  
  endif()

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
