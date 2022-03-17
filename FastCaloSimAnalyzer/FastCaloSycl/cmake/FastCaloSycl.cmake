# Copyright (C) 2002-2020 CERN for the benefit of the ATLAS collaboration

add_definitions(-DUSE_SYCL)

set(SYCL_DEBUG_INFO
  OFF CACHE BOOL "Enable debugging information output")
set(ENABLE_SYCL_TESTS
  OFF CACHE BOOL "Add tests to build")
set(SYCL_PROFILING_INFO
  OFF CACHE BOOL "Enable profiling information output")

if (SYCL_DEBUG_INFO)
  add_definitions(-DSYCL_DEBUG)
endif()

if (SYCL_PROFILING_INFO)
  add_definitions(-DSYCL_PROFILING)
endif()

set(SYCL_TARGET_DEFAULT
  OFF CACHE BOOL "Use the default device for simulation")
set(SYCL_TARGET_CPU
  OFF CACHE BOOL "Use the CPU device for simulation")
set(SYCL_TARGET_GPU
  OFF CACHE BOOL "Use the GPU device for simulation")
set(SYCL_TARGET_CUDA
  OFF CACHE BOOL "Enable CUDA backend for simulation")
set(SYCL_TARGET_HIP
  OFF CACHE BOOL "Enable HIP backend for simulation")

if (SYCL_TARGET_CUDA)
  set(INTERFACE_COMPILE_OPTIONS -fsycl -fsycl-targets=nvptx64-nvidia-cuda-sycldevice -fsycl-unnamed-lambda)
  set(INTERFACE_LINK_OPTIONS -fsycl -fsycl-targets=nvptx64-nvidia-cuda-sycldevice)
  set(CMAKE_CXX_FLAGS
    "-O2 -fsycl -fsycl-targets=nvptx64-nvidia-cuda-sycldevice -fsycl-unnamed-lambda -Wno-unknown-cuda-version")
  set_target_properties(${FastCaloSycl_LIB} PROPERTIES
  INTERFACE_COMPILE_OPTIONS "${INTERFACE_COMPILE_OPTIONS}"
  INTERFACE_LINK_OPTIONS "${INTERFACE_LINK_OPTIONS}")
  add_definitions(-DSYCL_TARGET_CUDA)
elseif(SYCL_TARGET_HIP)
  find_package(hipSYCL CONFIG REQUIRED)
  add_compile_options("-Wno-ignored-attributes --hipsycl-targets=hip:gfx900") # Silence HIP warnings
  # list(APPEND CMAKE_CXX_FLAGS "--hipsycl-targets=hip:gfx900")
  add_definitions(-DSYCL_TARGET_HIP)
else()
  set(CMAKE_CXX_FLAGS
  "-g -O2 -fsycl -Wno-unknown-cuda-version -fno-sycl-early-optimizations")
  if (SYCL_TARGET_DEFAULT)
    add_definitions(-DSYCL_TARGET_DEFAULT)
    message(STATUS " ${PROJECT_NAME} targeting default SYCL device for Geo")
  elseif (SYCL_TARGET_CPU)
    add_definitions(-DSYCL_TARGET_CPU)
    message(STATUS " ${PROJECT_NAME} targeting CPU SYCL device for Geo")
  elseif (SYCL_TARGET_GPU)
    add_definitions(-DSYCL_TARGET_GPU)
    message(STATUS " ${PROJECT_NAME} targeting GPU SYCL device for Geo")
  else()
    message(STATUS " ${PROJECT_NAME} targeting host SYCL device for Geo")
  endif()
endif()

if (NOT INPUT_PATH STREQUAL "")
  message(STATUS "Overriding all inputs path to '${INPUT_PATH}'")
  add_definitions(-DFCS_INPUT_PATH=\"${INPUT_PATH}\")
endif()
