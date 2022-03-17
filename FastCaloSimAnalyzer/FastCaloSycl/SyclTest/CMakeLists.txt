# Copyright (C) 2002-2020 CERN for the benefit of the ATLAS collaboration

cmake_minimum_required(VERSION 3.10)

project(SyclTest)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY
  ${CMAKE_TEST_OUTPUT_DIRECTORY}/FastCaloSycl)

add_executable(Histo_test)
target_sources(
  Histo_test
  PRIVATE Histo_test.cxx)
target_include_directories(
  Histo_test
  PRIVATE SyclCommon
          ${${AthenaStandalone_LIB}_Includes})
target_link_libraries(
  Histo_test
  PRIVATE SyclCommon
          ${AthenaStandalone_LIB})

add_executable(GeoRegion_test)
target_sources(
  GeoRegion_test
  PRIVATE GeoRegion_test.cxx)
target_include_directories(
  GeoRegion_test
  PRIVATE ${${AthenaStandalone_LIB}_Includes}
          ${${FastCaloSimCommon_LIB}_Includes})
target_link_libraries(
  GeoRegion_test
  PRIVATE SyclGeo
          ${AthenaStandalone_LIB}
          ${FastCaloSimCommon_LIB})

add_executable(Geo_test)
target_sources(
  Geo_test
  PRIVATE Geo_test.cxx)
target_include_directories(
  Geo_test
  PRIVATE ${${AthenaStandalone_LIB}_Includes}
          ${${FastCaloSimCommon_LIB}_Includes})
target_link_libraries(
  Geo_test
  PRIVATE SyclGeo
          ${AthenaStandalone_LIB}
          ${FastCaloSimCommon_LIB})

include(SyclRng)
add_executable(SimHitRng_test)
target_sources(
  SimHitRng_test
  PRIVATE SimHitRng_test.cxx)
target_link_libraries(
  SimHitRng_test
  PRIVATE SyclRng
          SyclCommon
          ${FastCaloSycl_LIB})
