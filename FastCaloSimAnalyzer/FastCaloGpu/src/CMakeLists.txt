# Copyright (C) 2002-2019 CERN for the benefit of the ATLAS collaboration

# "First-class" CUDA support needs at least CMake 3.10.
cmake_minimum_required( VERSION 3.10 )
cmake_policy(SET CMP0074 NEW)

# don't check for CUDA if either Kokkos or alpaka is used and the backend is something else
set(FIND_CUDA ON)

if(USE_KOKKOS)
  set(FIND_CUDA OFF)
endif()

if(USE_ALPAKA)
  if(${alpaka_ACC_GPU_CUDA_ENABLE} STREQUAL "OFF")
    set(FIND_CUDA OFF)
  endif()
endif()

if(USE_HIP)
  set(FIND_CUDA OFF)
endif()
 

if(ENABLE_OMPGPU)
  string(STRIP ${CMAKE_CXX_FLAGS} OMP_OFFLOAD_TARGET)
  string(FIND ${CMAKE_CXX_FLAGS} "gfx" OMP_OFFLOAD_TARGET_AMD)
  string(FIND ${CMAKE_CXX_FLAGS} "sm_" OMP_OFFLOAD_TARGET_NVIDIA)
  if(OMP_OFFLOAD_TARGET_NVIDIA GREATER 0)
    message(STATUS "OMP_OFFLOAD_TARGET NVIDIA" )
  elseif(OMP_OFFLOAD_TARGET_AMD GREATER 0)
    set(FIND_CUDA OFF)
    message(STATUS "OMP_OFFLOAD_TARGET AMD" )
  else()
    if($ENV{OMP_TARGET_OFFLOAD} MATCHES "disabled")
      set(FIND_CUDA OFF)
    else()
      message(FATAL_ERROR "!! Please specify OpenMP offload target via -DCMAKE_CXX_FLAGS=\"--offload-arch=gfx<>|sm_<>\" or set environment var OMP_TARGET_OFFLOAD=disabled")
    endif()
  endif()
endif()
 
if(FIND_CUDA)
  find_package(CUDA REQUIRED)
  enable_language( CUDA )
  set(CUDA_LIBRARIES PUBLIC ${CUDA_LIBRARIES})
endif()

# Sources

if(USE_STDPAR)
  set(FastCaloGpu_Srcs GeoLoadGpu.cxx KernelWrapper_sp.cxx gpuQ.cu Rand4Hits.cu Rand4Hits_sp.cxx )
elseif(USE_KOKKOS)
  set(FastCaloGpu_Srcs GeoLoadGpu.cxx KernelWrapper_kk.cxx DEV_BigMem_kk.cxx)
elseif(ENABLE_OMPGPU)
  # Add OpenMP
  find_package(OpenMP)
  if(OPENMP_FOUND)
    set(OpenMP_OPT_FLAGS "${OpenMP_OPT_FLAGS} -fopenmp-cuda-mode")
    set(OpenMP_OPT_FLAGS "${OpenMP_OPT_FLAGS} -foffload-lto")
    set(OpenMP_OPT_FLAGS "${OpenMP_OPT_FLAGS} -fopenmp-assume-no-thread-state")
    set(OpenMP_OPT_RMRKS "-Rpass=openmp-opt -Rpass-analysis=openmp-opt -Rpass-missed=openmp-opt " )
    set(OpenMP_FLAGS "-fopenmp -lomp -lomptarget")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_FLAGS} ${OpenMP_OPT_FLAGS} ${OpenMP_OPT_RMRKS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_FLAGS} ${OpenMP_OPT_FLAGS} ${OpenMP_OPT_RMRKS}")
  else()
    message(WARNING "Configuring with OpenMP GPU but OpenMP is not found!")
  endif()
  set(FastCaloGpu_Srcs KernelWrapper_omp.cxx gpuQ.cxx CaloGpuGeneral.cxx DEV_BigMem_omp.cxx )
elseif(USE_ALPAKA)
  set(FastCaloGpu_Srcs CaloGpuGeneral.cxx KernelWrapper_al.cxx Rand4Hits_al.cxx )
elseif(USE_HIP)
  
  # Define ROCM_PATH if not defined
  if (NOT DEFINED ENV{ROCM_PATH})
    message(WARNING "Environment variable ROCM_PATH not set! Using default /opt/rocm/")
    set(ROCM_PATH "/opt/rocm" CACHE STRING "Default ROCM installation directory.")
  else()
    set(ROCM_PATH $ENV{ROCM_PATH})
  endif()
  include_directories( ${ROCM_PATH}/hip/include )

  if ( ${HIP_TARGET} STREQUAL "AMD" )
    message(STATUS "Using AMD HIP backend")
    set(CMAKE_HIP_ARCHITECTURES "gfx90a;gfx906;gfx908")
  elseif( ${HIP_TARGET} STREQUAL "NVIDIA" )
    message(STATUS "Using NVIDIA HIP backend")
    find_package(CUDAToolkit REQUIRED)
    set(CMAKE_CUDA_ARCHITECTURES "70;75;80;86")
    set(CMAKE_HIP_ARCHITECTURES "gfx90a;gfx906;gfx908")
  else()
    message(FATAL_ERROR "unknown HIP_TARGET=${HIP_TARGET}. Must be either AMD or NVIDIA")
  endif()
    
  set(FastCaloGpu_Srcs GeoLoadGpu.cxx DEV_BigMem_hip.cxx KernelWrapper_hip.cxx gpuQ_hip.cxx Rand4Hits_hip.cxx )

else()
  set(FastCaloGpu_Srcs GeoLoadGpu.cxx gpuQ.cu CaloGpuGeneral.cxx  KernelWrapper_cu.cu DEV_BigMem.cu Rand4Hits.cu )
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

# Define and build the library
message(STATUS "Creating library target '${FastCaloGpu_LIB}'")
if(USE_ALPAKA)
  alpaka_add_library(${FastCaloGpu_LIB} SHARED ${FastCaloGpu_Srcs}  )
else()
  add_library(${FastCaloGpu_LIB} SHARED ${FastCaloGpu_Srcs}  )
endif()

target_include_directories(${FastCaloGpu_LIB} PRIVATE ../FastCaloGpu/ } )

if(USE_ALPAKA)
  target_link_libraries(${FastCaloGpu_LIB} PUBLIC ${CUDA_curand_LIBRARY} alpaka::alpaka)
  target_compile_definitions(${FastCaloGpu_LIB} PRIVATE ${FCS_CommonDefinitions})
elseif(USE_HIP)
  target_compile_definitions(${FastCaloGpu_LIB} PRIVATE ${FCS_CommonDefinitions})
  target_include_directories(${FastCaloGpu_LIB} PRIVATE ../FastCaloGpu/ } )
  
  if ( ${HIP_TARGET} STREQUAL "AMD" )
    target_compile_definitions(${FastCaloGpu_LIB} PUBLIC -DHIP_TARGET_AMD)
  elseif( ${HIP_TARGET} STREQUAL "NVIDIA" )
    target_compile_definitions(${FastCaloGpu_LIB} PUBLIC -DHIP_TARGET_NVIDIA)
    target_link_libraries(${FastCaloGpu_LIB} PUBLIC CUDA::cudart)
  endif()
elseif(ENABLE_OMPGPU)
  if(OMP_OFFLOAD_TARGET_NVIDIA GREATER 0)
    target_compile_definitions(${FastCaloGpu_LIB} PUBLIC -DOMP_OFFLOAD_TARGET_NVIDIA)
    target_link_libraries(${FastCaloGpu_LIB} PUBLIC ${CUDA_curand_LIBRARY} ${CUDA_nvToolsExt_LIBRARY})
  endif()
  if(OMP_OFFLOAD_TARGET_AMD GREATER 0)
    target_compile_definitions(${FastCaloGpu_LIB} PUBLIC -D__HIP_PLATFORM_AMD__)
    target_compile_definitions(${FastCaloGpu_LIB} PUBLIC -DOMP_OFFLOAD_TARGET_AMD)
    find_package(HIP REQUIRED) 
    target_include_directories(${FastCaloGpu_LIB} PRIVATE ${ROCM_PATH}/rocrand/include )
    target_include_directories(${FastCaloGpu_LIB} PRIVATE ${ROCM_PATH}/include )
    target_link_libraries(${FastCaloGpu_LIB}  PUBLIC ${ROCM_PATH}/lib/librocrand.so)
    target_link_libraries(${FastCaloGpu_LIB}  PUBLIC ${ROCM_PATH}/lib/libamdhip64.so)
  endif()
else()
  target_link_libraries(${FastCaloGpu_LIB} PUBLIC ${CUDA_curand_LIBRARY} ${CUDA_nvToolsExt_LIBRARY})
endif()

if(USE_STDPAR)
  target_link_libraries(${FastCaloGpu_LIB} PUBLIC  ${CUDA_LIBRARIES} )
  target_compile_definitions(${FastCaloGpu_LIB} PRIVATE -DUSE_STDPAR -DSTDPAR_TARGET=${STDPAR_TARGET} )

  target_compile_options(${FastCaloGpu_LIB} PRIVATE $<$<COMPILE_LANG_AND_ID:CXX,GNU>:
    ${STDPAR_DIRECTIVE}> )
  if( ${STDPAR_TARGET} STREQUAL "gpu" )
    target_compile_options(${FastCaloGpu_LIB} PRIVATE $<$<COMPILE_LANG_AND_ID:CXX,GNU>: -cuda -g -gpu=managed:intercept > )
  endif()
    
  target_link_options(${FastCaloGpu_LIB} PRIVATE ${STDPAR_DIRECTIVE})
elseif(USE_KOKKOS)
  target_link_libraries(${FastCaloGpu_LIB} PUBLIC  Kokkos::kokkos)
  target_compile_definitions(${FastCaloGpu_LIB} PRIVATE -DUSE_KOKKOS )
  if (CMAKE_BUILD_TYPE MATCHES Debug)
    target_compile_options(${FastCaloGpu_LIB} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>: -g -G -O0 > )
  endif()
else()
  if (CMAKE_BUILD_TYPE MATCHES Debug)
    target_compile_options(${FastCaloGpu_LIB} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>: -g -G -O0 > )
  endif()
endif()

if(RNDGEN_CPU)
  message(STATUS "Will generate random numbers on CPU")
  target_compile_definitions(${FastCaloGpu_LIB} PRIVATE -DRNDGEN_CPU )
elseif(RNDGEN_OMP)
  message(STATUS "Will generate random numbers using Portable OpenMP RNG Library")
  target_compile_definitions(${FastCaloGpu_LIB} PRIVATE -DRNDGEN_OMP )
  if(NOT DEFINED OMPRNG_HOME)
    include(FetchContent)
    set(FETCHCONTENT_QUIET OFF)
    FetchContent_Declare(
    Portable-OpenMP-RNG
    GIT_REPOSITORY https://github.com/GKNB/test-benchmark-OpenMP-RNG
    GIT_TAG        origin/main
    )
    FetchContent_Populate(Portable-OpenMP-RNG)
    include_directories(${CMAKE_BINARY_DIR}/_deps/portable-openmp-rng-src)
  else()
    include_directories(${OMPRNG_HOME})
    include_directories(${OMPRNG_HOME}/implementation)
  endif()
  if(ARCH_CUDA)
    target_compile_definitions(${FastCaloGpu_LIB} PRIVATE -DARCH_CUDA )
    target_link_libraries(${FastCaloGpu_LIB} PRIVATE ${CUDA_curand_LIBRARY} ${CUDA_nvToolsExt_LIBRARY})
  elseif(ARCH_HIP)
    target_compile_definitions(${FastCaloGpu_LIB} PRIVATE -DARCH_HIP )
    target_include_directories(${FastCaloGpu_LIB} PRIVATE ${ROCM_PATH}/rocrand/include )
  elseif(USE_RANDOM123)
    if(NOT DEFINED RANDOM123_HOME)
      include(FetchContent)
      set(FETCHCONTENT_QUIET OFF)
      FetchContent_Declare(
      random123
      GIT_REPOSITORY https://github.com/DEShawResearch/random123
      GIT_TAG        origin/main
      )
      FetchContent_Populate(random123)
      include_directories(${CMAKE_BINARY_DIR}/_deps/random123-src/include )
    else()
      include_directories(${RANDOM123_HOME}/include/)
    endif()
    target_compile_definitions(${FastCaloGpu_LIB} PRIVATE -DUSE_RANDOM123 )
  endif()
else()
  if(USE_HIP)
    if ( ${HIP_TARGET} STREQUAL "AMD" )
      target_include_directories(${FastCaloGpu_LIB} PRIVATE ${ROCM_PATH}/hiprand/include )
      target_include_directories(${FastCaloGpu_LIB} PRIVATE ${ROCM_PATH}/rocrand/include )
      target_link_libraries(${FastCaloGpu_LIB}  PUBLIC ${ROCM_PATH}/lib/libhiprand.so)
    elseif( ${HIP_TARGET} STREQUAL "NVIDIA" )
      target_link_libraries(${FastCaloGpu_LIB} PUBLIC ${CUDA_curand_LIBRARY} ${CUDA_nvToolsExt_LIBRARY})
    endif()
  elseif(ENABLE_OMPGPU)
    if(ENV{OMP_TARGET_OFFLOAD} MATCHES "disabled")
      message(FATAL_ERROR "when OMP_TARGET_OFFLOAD disabled, RNDGEN_CPU must be ON")
    endif()
  endif()
endif()

if(DUMP_HITCELLS)
  target_compile_definitions(${FastCaloGpu_LIB} PRIVATE -DDUMP_HITCELLS )
endif()

if(USE_ATOMIC_ADD)
  target_compile_definitions(${FastCaloGpu_LIB} PRIVATE -DUSE_ATOMICADD )
endif()
  
# This is a hack to prevent the exposure of alpaka definitions outside
# FastCaloGpu library via HostDevDef.h
if(USE_ALPAKA)
  target_compile_definitions(${FastCaloGpu_LIB} PRIVATE -DALPAKA_LOCAL )
endif()

if(ENABLE_OMPGPU)
  target_compile_definitions(${FastCaloGpu_LIB} PRIVATE -DUSE_OMPGPU )
  target_link_libraries(${FastCaloGpu_LIB} PUBLIC  ${CUDA_LIBRARIES} )
endif()

# Install library
install(TARGETS ${FastCaloGpu_LIB}
  DESTINATION ${CMAKE_INSTALL_LIBDIR}
)





