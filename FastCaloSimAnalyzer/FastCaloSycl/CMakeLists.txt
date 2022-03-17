# Copyright (C) 2002-2020 CERN for the benefit of the ATLAS collaboration

cmake_minimum_required(VERSION 3.10)

project(FastCaloSycl)

add_library(${FastCaloSycl_LIB} SHARED)

set(CMAKE_MODULE_PATH
  ${CMAKE_CURRENT_LIST_DIR}/cmake
)

include(FastCaloSycl)

target_sources(
  ${FastCaloSycl_LIB}
  PUBLIC ${CMAKE_CURRENT_LIST_DIR}/include/FastCaloSycl/SimEvent.h
         ${CMAKE_CURRENT_LIST_DIR}/include/FastCaloSycl/SimHit.h)

# target_sources(
#   ${FastCaloSycl_LIB}
#   PRIVATE src/SimEvent.cxx)

set_target_properties(FastCaloSycl PROPERTIES LINKER_LANGUAGE CXX)

target_link_libraries(
  ${FastCaloSycl_LIB}
  PRIVATE SyclGeo
          SyclRng
          SyclCommon)

target_include_directories(
  ${FastCaloSycl_LIB}
  PUBLIC ${CMAKE_CURRENT_LIST_DIR}/include)

install(
  TARGETS ${FastCaloSycl_LIB}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

add_subdirectory(SyclGeo)
add_subdirectory(SyclRng)
add_subdirectory(SyclCommon)

if (ENABLE_SYCL_TESTS)
  add_subdirectory(SyclTest)
endif()
