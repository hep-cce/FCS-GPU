# Copyright (C) 2002-2020 CERN for the benefit of the ATLAS collaboration

cmake_minimum_required(VERSION 3.10)

add_library(SyclRng SHARED)

include(SyclRng)

set(LINK_LIBS
  mkl_sycl mkl_intel_ilp64 mkl_tbb_thread mkl_core tbb)

target_sources(
  SyclRng
  PUBLIC ${CMAKE_CURRENT_LIST_DIR}/include/SyclRng/SimHitRng.h)

target_sources(
  SyclRng
  PRIVATE src/SimHitRng.cxx)

target_include_directories(
  SyclRng
  PUBLIC ${CMAKE_CURRENT_LIST_DIR}/include)

target_link_libraries(
  SyclRng
  PRIVATE SyclCommon
          ${LINK_LIBS})

install(TARGETS SyclRng
  DESTINATION ${CMAKE_INSTALL_LIBDIR})
