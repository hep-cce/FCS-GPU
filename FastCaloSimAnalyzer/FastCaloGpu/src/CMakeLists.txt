# Copyright (C) 2002-2019 CERN for the benefit of the ATLAS collaboration

# "First-class" CUDA support needs at least CMake 3.10.
cmake_minimum_required( VERSION 3.10 )
cmake_policy(SET CMP0074 NEW)

# Set up the "modern" usage of CUDA with CMake.
find_package(CUDA REQUIRED)
enable_language( CUDA )

#Need to do other way.
# set(CMAKE_CUDA_FLAGS "-arch=sm_75")
# Sources

if(USE_STDPAR)
  set(FastCaloGpu_Srcs GeoLoadGpu.cxx KernelWrapper_sp.cxx gpuQ.cu Rand4Hits.cu )
#  set(FastCaloGpu_Srcs gpuQ.cu CaloGpuGeneral.cxx  KernelWrapper.cu Rand4Hits.cu ) 
elseif(USE_KOKKOS)
  set(FastCaloGpu_Srcs GeoLoadGpu.cxx KernelWrapper_kk.cxx )
else()
  set(FastCaloGpu_Srcs gpuQ.cu CaloGpuGeneral.cxx  KernelWrapper_cu.cu Rand4Hits.cu ) 
endif()

# # Global include is needed for dictionary generation to work
#include_directories(../../)

# Add dictionary dependencies
#fcs_dictionary_dependency(${FastCaloGpu_LIB})

# ROOT dictionary
#root_generate_dictionary(
#  ${FastCaloGpu_LIB}Dict

#  MODULE ${FastCaloGpu_LIB}
#  LINKDEF LinkDef.h
#)

#This seems to be a hack to avoid a cmake error 
set(CUDA_LIBRARIES PUBLIC  ${CUDA_LIBRARIES})

# Define and build the library
message(STATUS "Creating library target '${FastCaloGpu_LIB}'")
add_library(${FastCaloGpu_LIB} SHARED ${FastCaloGpu_Srcs}  )

target_include_directories(${FastCaloGpu_LIB} PRIVATE ../FastCaloGpu/  )

target_link_libraries(${FastCaloGpu_LIB}  PUBLIC ${CUDA_curand_LIBRARY})
if(USE_STDPAR)
  #  target_link_libraries(${FastCaloGpu_LIB} PUBLIC  Kokkos::kokkos)
  target_link_libraries(${FastCaloGpu_LIB} PUBLIC  ${CUDA_LIBRARIES} )
  target_compile_definitions(${FastCaloGpu_LIB} PRIVATE -DUSE_STDPAR )
  # target_compile_options(${FastCaloGpu_LIB} PRIVATE -stdpar=gpu )
endif()

if(RNDGEN_CPU)
  message(STATUS "Will generate random numbers on CPU")
  target_compile_definitions(${FastCaloGpu_LIB} PRIVATE -DRNDGEN_CPU )
endif()

# Install library
install(TARGETS ${FastCaloGpu_LIB}
  DESTINATION ${CMAKE_INSTALL_LIBDIR}
)





