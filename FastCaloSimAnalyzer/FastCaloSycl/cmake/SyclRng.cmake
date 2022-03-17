# Copyright (C) 2002-2020 CERN for the benefit of the ATLAS collaboration

set(CMAKE_CXX_FLAGS
  "-g -O2 -fsycl -Wno-unknown-cuda-version")

set(SYCL_RNG_DEFAULT
  OFF CACHE BOOL "Use default device for RNG")
set(SYCL_RNG_CPU
  OFF CACHE BOOL "Use CPU device for RNG")
set(SYCL_RNG_GPU
  OFF CACHE BOOL "Use GPU device for RNG")
set(SYCL_RNG_CUDA
  OFF CACHE BOOL "Use CUDA device for RNG")
set(SYCL_RNG_HIP
  OFF CACHE BOOL "Use HIP device for RNG")

if (SYCL_RNG_DEFAULT)
  message(STATUS " ${PROJECT_NAME} targeting default SYCL device for RNG")
  add_definitions(-DSYCL_RNG_DEFAULT)
elseif (SYCL_RNG_CPU)
  message(STATUS " ${PROJECT_NAME} targeting CPU SYCL device for RNG")
  add_definitions(-DSYCL_RNG_CPU)
elseif (SYCL_RNG_GPU)
  message(STATUS " ${PROJECT_NAME} targeting GPU SYCL device for RNG")
  add_definitions(-DSYCL_RNG_GPU)
elseif (SYCL_RNG_CUDA)
  message(STATUS " ${PROJECT_NAME} targeting CUDA SYCL device for RNG")
  add_definitions(-DSYCL_RNG_CUDA)
elseif (SYCL_RNG_HIP)
  message(STATUS " ${PROJECT_NAME} targeting HIP SYCL device for RNG")
  find_package(hipSYCL REQUIRED)
  add_compile_options("-Wno-ignored-attributes --hipsycl-targets=hip:gfx900") # Silence HIP warnings
  # add_compile_options("-Wno-ignored-attributes") # Silence HIP warnings
  # list(APPEND CMAKE_CXX_FLAGS "--hipsycl-targets=hip:gfx900")
  # set(INTERFACE_COMPILE_OPTIONS "-fsycl -Wno-ignored-attributes --hipsycl-targets=hip:gfx900 -fsycl-unnamed-lambda")
  # set(INTERFACE_LINK_OPTIONS "-fsycl -Wno-ignored-attributes --hipsycl-targets=hip:gfx900")
  add_definitions(-DSYCL_TARGET_HIP)
else()
  message(STATUS " ${PROJECT_NAME} targeting host SYCL device for RNG")
endif()
