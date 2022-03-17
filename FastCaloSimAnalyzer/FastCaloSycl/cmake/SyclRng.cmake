# Copyright (C) 2002-2020 CERN for the benefit of the ATLAS collaboration

set(CMAKE_CXX_FLAGS
  "-g -O2 -fsycl -std=c++17 -Wno-unknown-cuda-version")

set(SYCL_RNG_DEFAULT
  OFF CACHE BOOL "Use default device for RNG")
set(SYCL_RNG_CPU
  OFF CACHE BOOL "Use CPU device for RNG")
set(SYCL_RNG_GPU
  OFF CACHE BOOL "Use GPU device for RNG")

if (SYCL_RNG_DEFAULT)
  message(STATUS " ${PROJECT_NAME} targeting default SYCL device for RNG")
  add_definitions(-DSYCL_RNG_DEFAULT)
elseif (SYCL_RNG_CPU)
  message(STATUS " ${PROJECT_NAME} targeting CPU SYCL device for RNG")
  add_definitions(-DSYCL_RNG_CPU)
elseif (SYCL_RNG_GPU)
  message(STATUS " ${PROJECT_NAME} targeting GPU SYCL device for RNG")
  add_definitions(-DSYCL_RNG_GPU)
else()
  message(STATUS " ${PROJECT_NAME} targeting host SYCL device for RNG")
endif()
