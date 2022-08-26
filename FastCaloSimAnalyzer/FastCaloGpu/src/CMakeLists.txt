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
  set(FastCaloGpu_Srcs GeoLoadGpu.cxx KernelWrapper_sp.cxx gpuQ.cu Rand4Hits.cu Rand4Hits_sp.cxx )
#  set(FastCaloGpu_Srcs gpuQ.cu CaloGpuGeneral.cxx  KernelWrapper.cu Rand4Hits.cu ) 
elseif(USE_KOKKOS)
  set(FastCaloGpu_Srcs GeoLoadGpu.cxx KernelWrapper_kk.cxx )
elseif(USE_ALPAKA)
  set(FastCaloGpu_Srcs gpuQ.cu CaloGpuGeneral.cxx KernelWrapper_al.cxx Rand4Hits_al.cxx )
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
if(USE_ALPAKA)
  alpaka_add_library(${FastCaloGpu_LIB} SHARED ${FastCaloGpu_Srcs}  )
else()
  add_library(${FastCaloGpu_LIB} SHARED ${FastCaloGpu_Srcs}  )
endif()

target_include_directories(${FastCaloGpu_LIB} PRIVATE ../FastCaloGpu/  )

if(USE_ALPAKA)
  target_link_libraries(${FastCaloGpu_LIB}  PUBLIC ${CUDA_curand_LIBRARY} alpaka::alpaka)
  target_compile_definitions(${FastCaloGpu_LIB} PRIVATE ${FCS_CommonDefinitions})
else()
  target_link_libraries(${FastCaloGpu_LIB}  PUBLIC ${CUDA_curand_LIBRARY} ${CUDA_nvToolsExt_LIBRARY})
endif()

if(USE_STDPAR)
  target_link_libraries(${FastCaloGpu_LIB} PUBLIC  ${CUDA_LIBRARIES} )
  target_compile_definitions(${FastCaloGpu_LIB} PRIVATE -DUSE_STDPAR -DSTDPAR_TARGET=${STDPAR_TARGET} )

  target_compile_options(${FastCaloGpu_LIB} PRIVATE $<$<COMPILE_LANG_AND_ID:CXX,GNU>:
    ${STDPAR_DIRECTIVE}> )
  if( ${STDPAR_TARGET} STREQUAL "gpu" )
    target_compile_options(${FastCaloGpu_LIB} PRIVATE $<$<COMPILE_LANG_AND_ID:CXX,GNU>: -cuda -g>)
  endif()
    
  target_link_options(${FastCaloGpu_LIB} PRIVATE ${STDPAR_DIRECTIVE})

endif()

if(RNDGEN_CPU)
  message(STATUS "Will generate random numbers on CPU")
  target_compile_definitions(${FastCaloGpu_LIB} PRIVATE -DRNDGEN_CPU )
endif()

if(DUMP_HITCELLS)
  target_compile_definitions(${FastCaloGpu_LIB} PRIVATE -DDUMP_HITCELLS )
endif()

if(USE_ATOMIC_ADD)
  target_compile_definitions(${FastCaloGpu_LIB} PRIVATE -DUSE_ATOMICADD )
endif()

# Install library
install(TARGETS ${FastCaloGpu_LIB}
  DESTINATION ${CMAKE_INSTALL_LIBDIR}
)





