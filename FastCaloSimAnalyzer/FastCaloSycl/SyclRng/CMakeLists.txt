# Copyright (C) 2002-2020 CERN for the benefit of the ATLAS collaboration

cmake_minimum_required(VERSION 3.10)

add_library(SyclRng SHARED)

include(SyclRng)

if (SYCL_RNG_CPU)
  set(MKL_RNG_LIB onemkl)
elseif (SYCL_RNG_GPU)
  set(MKL_RNG_LIB onemkl)
endif()

#set(LINK_LIBS onemkl)
set(LINK_LIBS OpenCL)

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

target_link_options(
  SyclRng PRIVATE -lsycl -qmkl )

install(TARGETS SyclRng
  DESTINATION ${CMAKE_INSTALL_LIBDIR})
