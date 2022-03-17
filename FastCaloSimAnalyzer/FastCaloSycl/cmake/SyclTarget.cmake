# Copyright (C) 2002-2020 CERN for the benefit of the ATLAS collaboration

set(SYCL_TARGET_DEFAULT
  OFF CACHE BOOL "Use the default device for simulation")
set(SYCL_TARGET_CPU
  OFF CACHE BOOL "Use the CPU device for simulation")
set(SYCL_TARGET_GPU
  OFF CACHE BOOL "Use the GPU device for simulation")
set(SYCL_TARGET_CUDA
  OFF CACHE BOOL "Enable CUDA backend for simulation")

if (SYCL_TARGET_CUDA)
  set(CMAKE_CXX_FLAGS
    "-g -O2 -fsycl -std=c++17 -fsycl-targets=nvptx64-nvidia-cuda-sycldevice -Wno-unknown-cuda-version")
  add_definitions(-DSYCL_TARGET_CUDA)
else()
  set(CMAKE_CXX_FLAGS
  "-g -O2 -fsycl -std=c++17 -Wno-unknown-cuda-version")
  if (SYCL_TARGET_DEFAULT)
    add_definitions(-DSYCL_TARGET_DEFAULT)
    message(STATUS " Targetting default SYCL device")
  elseif (SYCL_TARGET_CPU)
    add_definitions(-DSYCL_TARGET_CPU)
    message(STATUS " Targetting CPU SYCL device")
  elseif (SYCL_TARGET_GPU)
    add_definitions(-DSYCL_TARGET_GPU)
    message(STATUS " Targetting GPU SYCL device")
  endif()
endif()
