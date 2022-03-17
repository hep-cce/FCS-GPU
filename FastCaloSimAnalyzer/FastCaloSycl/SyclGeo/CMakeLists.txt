# Copyright (C) 2002-2020 CERN for the benefit of the ATLAS collaboration

cmake_minimum_required(VERSION 3.10)

add_library(SyclGeo SHARED)

target_sources(
  SyclGeo
  PUBLIC ${CMAKE_CURRENT_LIST_DIR}/include/SyclGeo/Geo.h
         ${CMAKE_CURRENT_LIST_DIR}/include/SyclGeo/GeoRegion.h)

target_sources(
  SyclGeo
  PRIVATE include/SyclGeo/GeoRegion.icc
          src/Geo.cxx
          src/GeoRegion.cxx)

target_include_directories(
  SyclGeo
  PUBLIC ${CMAKE_CURRENT_LIST_DIR}/include)

target_include_directories(
  SyclGeo
  PUBLIC ${${AthenaStandalone_LIB}_Includes})

target_include_directories(
  SyclGeo
  PRIVATE ${${FastCaloSimCommon_LIB}_Includes})

target_link_libraries(
  SyclGeo
  PUBLIC SyclCommon)

target_link_libraries(
  SyclGeo
  PRIVATE SyclCommon
          ${AthenaStandalone_LIB}
          ${FastCaloSimCommon_LIB})

install(TARGETS SyclGeo
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
)
