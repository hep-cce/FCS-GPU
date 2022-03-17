# Copyright (C) 2002-2020 CERN for the benefit of the ATLAS collaboration

cmake_minimum_required(VERSION 3.10)

add_library(SyclCommon SHARED)

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

install(
  TARGETS SyclCommon
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
)
