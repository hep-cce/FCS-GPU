# Copyright (C) 2002-2019 CERN for the benefit of the ATLAS collaboration

# "First-class" CUDA support needs at least CMake 3.10.
cmake_minimum_required( VERSION 3.10 )

# use HIP
find_package(HIP REQUIRED)
set(FastCaloGpu_Srcs GeoLoadGpu.cpp )

# # Global include is needed for dictionary generation to work
#include_directories(../../)

## for HIP
include_directories( /opt/rocm/hip/include )
add_compile_definitions(__HIP_PLATFORM_AMD__ HIP_PLATFORM_AMD __HIP_PLATFORM_HCC__ HIP_PLATFORM_HCC)



# Add dictionary dependencies
#fcs_dictionary_dependency(${FastCaloGpu_LIB})

# ROOT dictionary
#root_generate_dictionary(
#  ${FastCaloGpu_LIB}Dict

#  MODULE ${FastCaloGpu_LIB}
#  LINKDEF LinkDef.h
#)

# Define and build the library
message(STATUS "Creating library target '${FastCaloGpu_LIB}'")
add_library(${FastCaloGpu_LIB} SHARED ${FastCaloGpu_Srcs}  )

target_include_directories(${FastCaloGpu_LIB} PRIVATE ../FastCaloGpu/ } )
target_include_directories(${FastCaloGpu_LIB} PRIVATE /opt/rocm/hiprand/include )
target_include_directories(${FastCaloGpu_LIB} PRIVATE /opt/rocm/rocrand/include )

# target_link_libraries(${FastCaloGpu_LIB}  PUBLIC /opt/rocm/hiprand/lib/libhiprand.so)
target_link_libraries(${FastCaloGpu_LIB}  PUBLIC /opt/rocm/lib/libhiprand.so)

if(RNDGEN_CPU)
  message(STATUS "Will generate random numbers on CPU")
  target_compile_definitions(${FastCaloGpu_LIB} PRIVATE -DRNDGEN_CPU )
endif()

if(DUMP_HITCELLS)
  target_compile_definitions(${FastCaloGpu_LIB} PRIVATE -DDUMP_HITCELLS )
endif()

# Install library
install(TARGETS ${FastCaloGpu_LIB}
  DESTINATION ${CMAKE_INSTALL_LIBDIR}
)





