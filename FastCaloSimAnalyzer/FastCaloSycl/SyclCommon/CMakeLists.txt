# Copyright (C) 2002-2020 CERN for the benefit of the ATLAS collaboration

cmake_minimum_required(VERSION 3.10)

add_library(SyclCommon SHARED)

include(SyclRng)

target_sources(
  SyclCommon
  PUBLIC ${CMAKE_CURRENT_LIST_DIR}/include/SyclCommon/Histo.h
          ${CMAKE_CURRENT_LIST_DIR}/include/SyclCommon/Props.h
         ${CMAKE_CURRENT_LIST_DIR}/include/SyclCommon/DeviceCommon.h)

target_sources(
  SyclCommon
  PRIVATE include/SyclCommon/Histo.icc
          src/Histo.cxx)

target_include_directories(
  SyclCommon
  PUBLIC ${CMAKE_CURRENT_LIST_DIR}/include)

target_include_directories(
  SyclCommon
  PRIVATE ${${FastCaloSimAnalyzer_LIB}_Includes}
          ${${AthenaStandalone_LIB}_Includes})

if (SYCL_TARGET_CUDA)
  set(INTERFACE_COMPILE_OPTIONS -fsycl -fsycl-targets=nvptx64-nvidia-cuda-sycldevice -fsycl-unnamed-lambda)
  set(INTERFACE_LINK_OPTIONS -fsycl -fsycl-targets=nvptx64-nvidia-cuda-sycldevice)
  # set(CMAKE_CXX_FLAGS
  #   "-O2 -fsycl -std=c++17 -fsycl-targets=nvptx64-nvidia-cuda-sycldevice -fsycl-unnamed-lambda -Wno-unknown-cuda-version")
  set_target_properties(SyclCommon PROPERTIES
  INTERFACE_COMPILE_OPTIONS "${INTERFACE_COMPILE_OPTIONS}"
  INTERFACE_LINK_OPTIONS "${INTERFACE_LINK_OPTIONS}")
elseif(SYCL_TARGET_HIP)
  message(WARNING "SyclCommon using HIP!!!!!")
  find_package(hipSYCL REQUIRED)
  # add_compile_options("-Wno-ignored-attributes --hipsycl-targets=hip:gfx900") # Silence HIP warnings
  # add_compile_options("-Wno-ignored-attributes") # Silence HIP warnings
  # list(APPEND CMAKE_CXX_FLAGS "--hipsycl-targets=hip:gfx900")
  # set(INTERFACE_COMPILE_OPTIONS "-fsycl -Wno-ignored-attributes --hipsycl-targets=hip:gfx900 -fsycl-unnamed-lambda")
  # set(INTERFACE_LINK_OPTIONS "-fsycl -Wno-ignored-attributes --hipsycl-targets=hip:gfx900")
  set_target_properties(SyclCommon PROPERTIES
    INTERFACE_COMPILE_OPTIONS "${INTERFACE_COMPILE_OPTIONS}"
    INTERFACE_LINK_OPTIONS "${INTERFACE_LINK_OPTIONS}")
  add_definitions(-DSYCL_TARGET_HIP)
endif()

install(
  TARGETS SyclCommon
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
)
