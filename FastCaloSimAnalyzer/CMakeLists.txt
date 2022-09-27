# Copyright (C) 2002-2018 CERN for the benefit of the ATLAS collaboration

cmake_minimum_required(VERSION 3.14 FATAL_ERROR)

# Add additional CMake macros and definitions
set(CMAKE_MODULE_PATH
  ${CMAKE_SOURCE_DIR}/FastCaloSimCommon/cmake
)

# Set project
project(FastCaloSimAnalyzer VERSION 1.0.0)

set(ENABLE_GPU OFF CACHE BOOL "Enable GPU")
set(ENABLE_OMPGPU OFF CACHE BOOL "Enable OMP GPU")
set(RNDGEN_CPU OFF CACHE BOOL "Generate Random Numbers on CPU")

if(USE_KOKKOS)
  find_package(Kokkos) 
endif()

include(FastCaloSim)
include(XRootD)

# Add OpenMP
find_package(OpenMP)
if(OPENMP_FOUND)
	##set(OpenMP_FLAGS "-Xopenmp-target=amdgcn -march=gfx906 -fopenmp -fopenmp-targets=amdgcn")
	##set(OpenMP_FLAGS "-Xopenmp-target -march=sm_70 -fopenmp -fopenmp-cuda-mode -fopenmp-targets=nvptx64 -fopenmp-target-debug=0x5" )
	##set(OpenMP_FLAGS "-Xopenmp-target -march=sm_70 -fopenmp -fopenmp-cuda-mode -fopenmp-targets=nvptx64")##lambda1
	set(OpenMP_FLAGS "-Xopenmp-target -march=sm_86 -fopenmp -fopenmp-cuda-mode -fopenmp-targets=nvptx64")##lambda2
	##set(OpenMP_FLAGS "-Xopenmp-target -march=sm_75 -fopenmp -fopenmp-cuda-mode -fopenmp-targets=nvptx64")##lambda4
	##set(OpenMP_FLAGS "-mp=gpu -cuda -Minfo=mp") ## for nvc++
	set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_FLAGS} -maxrregcount=64")
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_FLAGS}")
else()
  message(WARNING "Configuring without OpenMP!")
endif()


set(PROJECT_SRC_DIR ${CMAKE_SOURCE_DIR}/Root)

add_subdirectory(FastCaloSimCommon/src FastCaloSimCommon)
add_subdirectory(FastCaloSimCommon/AthenaBuild AthenaBuild)
add_subdirectory(EnergyParametrization/src EnergyParametrization)
add_subdirectory(Root)
if(ENABLE_GPU) 
  add_subdirectory(FastCaloGpu/src FastCaloGpu)
elseif(ENABLE_OMPGPU) 
  add_subdirectory(FastCaloGpu/src FastCaloGpu)
endif() 

add_subdirectory(macro)
