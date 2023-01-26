# Copyright (C) 2002-2018 CERN for the benefit of the ATLAS collaboration

cmake_minimum_required(VERSION 3.8 FATAL_ERROR)

# Add additional CMake macros and definitions
set(CMAKE_MODULE_PATH
  ${CMAKE_SOURCE_DIR}/FastCaloSimCommon/cmake
)

# Set project
project(FastCaloSimAnalyzer VERSION 1.0.0)

set(ENABLE_GPU OFF CACHE BOOL "Enable GPU")
set(ENABLE_GPU OFF CACHE BOOL "Enable OMPTARGET GPU")

# Add OpenMP
find_package(OpenMP)
if(OPENMP_FOUND)
	set(OpenMP_OPT_FLAGS "${OpenMP_OPT_FLAGS} -fopenmp-cuda-mode")
	set(OpenMP_OPT_FLAGS "${OpenMP_OPT_FLAGS} -foffload-lto")
	set(OpenMP_OPT_FLAGS "${OpenMP_OPT_FLAGS} -fopenmp-assume-no-thread-state")
	#set(OpenMP_OPT_FLAGS "${OpenMP_OPT_FLAGS} -fopenmp-assume-no-nested-parallelism")
	#set(OpenMP_OPT_FLAGS "${OpenMP_OPT_FLAGS} -maxrregcount=64")
	set(OpenMP_OPT_RMRKS "-Rpass=openmp-opt -Rpass-analysis=openmp-opt -Rpass-missed=openmp-opt " )
	##set(OpenMP_DBG_FLAGS "-fopenmp-target-debug=0x5" )
	set(OpenMP_FLAGS "-fopenmp --offload-arch=sm_86 -lomp") ##lambda2
	#set(OpenMP_FLAGS "-fopenmp --offload-arch=gfx906") ##lambda3
	##set(OpenMP_FLAGS "-mp=gpu -gpu=cc86 -cuda -Minfo=mp") ## for nvc++
	set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_FLAGS} ${OpenMP_OPT_FLAGS} ${OpenMP_OPT_RMRKS}")
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_FLAGS} ${OpenMP_OPT_FLAGS} ${OpenMP_OPT_RMRKS}")
else()
  message(WARNING "Configuring without OpenMP!")
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
elseif(ENABLE_OMPGPU)
  add_subdirectory(FastCaloGpu/src FastCaloGpu)
endif() 
add_subdirectory(macro)
