# Copyright (C) 2002-2019 CERN for the benefit of the ATLAS collaboration

# "First-class" CUDA support needs at least CMake 3.10.
cmake_minimum_required( VERSION 3.17 )
cmake_policy(SET CMP0074 NEW)

set(FIND_CUDA ON)


# don't check for CUDA w/ Kokkos incase backend is something else
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

if(USE_STDPAR)
  if ( ${STDPAR_TARGET} STREQUAL "gpu" )
    find_package(NVHPC REQUIRED COMPONENTS MATH)
    set(CMAKE_THREAD_LIBS_INIT "-lpthread")
    set(CMAKE_HAVE_THREADS_LIBRARY 1)
    set(CMAKE_USE_WIN32_THREADS_INIT 0)
    set(CMAKE_USE_PTHREADS_INIT 1)
    set(THREADS_PREFER_PTHREAD_FLAG ON)
    set(FIND_CUDA ON)
  endif()
endif()

if(FIND_CUDA)
  find_package(CUDAToolkit REQUIRED)
  if(TARGET CUDA::curand)
    set(CURAND_LIB CUDA::curand)
  else()
    set(CURAND_LIB "curand")
  endif()

  if(TARGET CUDA::nvToolsExt)
    set(NVTOOLSEXT_LIB CUDA::nvToolsExt)
  else()
    set(NVTOOLSEXT_LIB "nvToolsExt")
  endif()

  if(TARGET CUDA::cudart)
    set(CUDART_LIB CUDA::cudart)
  else()
    set(CUDART_LIB "cudart")
  endif()

  if(TARGET CUDA::cuda_driver)
    set(CUDA_DRIVER_LIB CUDA::cuda_driver)
  else()
    set(CUDA_DRIVER_LIB "cuda_driver")
  endif()

  enable_language( CUDA )
  set(CUDA_LIBRARIES PUBLIC  ${CUDA_LIBRARIES})
endif()

if(ENABLE_OMPGPU)
  # Add OpenMP
  find_package(OpenMP)
  if(OPENMP_FOUND)
    set(OpenMP_OPT_FLAGS "${OpenMP_OPT_FLAGS} -fopenmp-cuda-mode")
    set(OpenMP_OPT_FLAGS "${OpenMP_OPT_FLAGS} -foffload-lto")
    set(OpenMP_OPT_FLAGS "${OpenMP_OPT_FLAGS} -fopenmp-assume-no-thread-state")
    #set(OpenMP_OPT_FLAGS "${OpenMP_OPT_FLAGS} -fopenmp-assume-no-nested-parallelism")
    #set(OpenMP_OPT_FLAGS "${OpenMP_OPT_FLAGS} -maxrregcount=64")
    set(OpenMP_OPT_RMRKS "-Rpass=openmp-opt -Rpass-analysis=openmp-opt -Rpass-missed=openmp-opt " )
    ##set(OpenMP_DBG_FLAGS "-fopenmp-target-debug=0x5" )
    set(OpenMP_FLAGS "-fopenmp --offload-arch=sm_86 -lomp") ##lambda2
    ##set(OpenMP_FLAGS "-mp=gpu -gpu=cc86 -cuda -Minfo=mp") ## for nvc++
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_FLAGS} ${OpenMP_OPT_FLAGS} ${OpenMP_OPT_RMRKS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_FLAGS} ${OpenMP_OPT_FLAGS} ${OpenMP_OPT_RMRKS}")
  else()
    message(WARNING "Configuring without OpenMP!")
  endif()
endif()


# Sources
if(USE_STDPAR)
  set(FastCaloGpu_Srcs GeoLoadGpu.cxx KernelWrapper_sp.cxx gpuQ.cu Rand4Hits.cu Rand4Hits_sp.cxx )
#  set(FastCaloGpu_Srcs gpuQ.cu CaloGpuGeneral.cxx  KernelWrapper.cu Rand4Hits.cu ) 
elseif(USE_KOKKOS)
  set(FastCaloGpu_Srcs GeoLoadGpu.cxx KernelWrapper_kk.cxx )
elseif(USE_ALPAKA)
  set(FastCaloGpu_Srcs CaloGpuGeneral.cxx KernelWrapper_al.cxx Rand4Hits_al.cxx )
elseif(ENABLE_OMPGPU)
  set(FastCaloGpu_Srcs KernelWrapper_omp.cxx)
elseif(USE_HIP)
  include_directories( /opt/rocm/hip/include )
  if ( ${HIP_TARGET} STREQUAL "AMD" )
    message(STATUS "    Using AMD HIP backend")
    add_compile_definitions(__HIP_PLATFORM_AMD__ HIP_PLATFORM_AMD)
  elseif( ${HIP_TARGET} STREQUAL "NVIDIA" )
    message(STATUS "    Using NVIDIA HIP backend")
    add_compile_definitions(__HIP_PLATFORM_NVIDIA__ HIP_PLATFORM_NVIDIA)
  else()
    message(FATAL_ERROR "unknown HIP_TARGET=${HIP_TARGET}. Must be either AMD or NVIDIA")
  endif()
  add_compile_definitions(__HIP_PLATFORM_HCC__ HIP_PLATFORM_HCC)
    
  set(FastCaloGpu_Srcs GeoLoadGpu.cxx KernelWrapper_hip.cxx gpuQ_hip.cxx Rand4Hits_hip.cxx )

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

# Define and build the library
message(STATUS "Creating library target '${FastCaloGpu_LIB}'")
if(USE_ALPAKA)
  alpaka_add_library(${FastCaloGpu_LIB} SHARED ${FastCaloGpu_Srcs}  )
else()
  add_library(${FastCaloGpu_LIB} SHARED ${FastCaloGpu_Srcs}  )
endif()

target_include_directories(${FastCaloGpu_LIB} PRIVATE ../FastCaloGpu/  )

if(USE_ALPAKA)
  target_link_libraries(${FastCaloGpu_LIB}  PUBLIC alpaka::alpaka ${CURAND_LIB})
  target_compile_definitions(${FastCaloGpu_LIB} PRIVATE ${FCS_CommonDefinitions})
elseif(USE_HIP)
  target_compile_definitions(${FastCaloGpu_LIB} PRIVATE ${FCS_CommonDefinitions})
  target_include_directories(${FastCaloGpu_LIB} PRIVATE ../FastCaloGpu/ } )
  target_include_directories(${FastCaloGpu_LIB} PRIVATE /opt/rocm/hiprand/include )
  target_include_directories(${FastCaloGpu_LIB} PRIVATE /opt/rocm/rocrand/include )

  target_link_libraries(${FastCaloGpu_LIB}  PUBLIC /opt/rocm/lib/libhiprand.so)
else()
  target_link_libraries(${FastCaloGpu_LIB}  PUBLIC ${NVTOOLSEXT_LIB} ${CURAND_LIB})
endif()

if(ENABLE_OMPGPU)
  target_compile_definitions(${FastCaloGpu_LIB} PRIVATE -DUSE_OMPGPU )
  target_link_libraries(${FastCaloGpu_LIB}  PUBLIC ${CUDA_curand_LIBRARY} ${CUDA_nvToolsExt_LIBRARY})
endif()

if(USE_KOKKOS)
  target_link_libraries(${FastCaloGpu_LIB} PUBLIC  Kokkos::kokkos)
  target_compile_definitions(${FastCaloGpu_LIB} PRIVATE -DUSE_KOKKOS )
  if (CMAKE_BUILD_TYPE MATCHES Debug)
    target_compile_options(${FastCaloGpu_LIB} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>: -g -G -O0 > )
  endif()
endif()

if(USE_STDPAR)  
  target_link_libraries(${FastCaloGpu_LIB} PUBLIC  ${CUDART_LIB} ${CUDA_DRIVER_LIB} )
  target_compile_definitions(${FastCaloGpu_LIB} PRIVATE -DUSE_STDPAR -DSTDPAR_TARGET=${STDPAR_TARGET} )

  #
  ### should remove the "GNU" stuff when can build whole project with nvc++
  #
  target_compile_options(${FastCaloGpu_LIB} PRIVATE $<$<COMPILE_LANG_AND_ID:CXX,GNU>:
    ${STDPAR_DIRECTIVE}> )
  target_compile_options(${FastCaloGpu_LIB} PRIVATE $<$<COMPILE_LANG_AND_ID:CXX,NVHPC>:
    ${STDPAR_DIRECTIVE}> )
  if( ${STDPAR_TARGET} STREQUAL "gpu" )
    target_compile_options(${FastCaloGpu_LIB} PRIVATE $<$<COMPILE_LANG_AND_ID:CXX,GNU>: -cuda -g>)
    target_compile_options(${FastCaloGpu_LIB} PRIVATE $<$<COMPILE_LANG_AND_ID:CXX,NVHPC>: -cuda -g>)

    target_include_directories(${FastCaloGpu_LIB} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:${NVHPC_ROOT_DIR}/math_libs/${NVHPC_CUDA_VERSION}/include>)    
    
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

# This is a hack to prevent the exposure of alpaka definitions outside
# FastCaloGpu library via HostDevDef.h
if(USE_ALPAKA)
  target_compile_definitions(${FastCaloGpu_LIB} PRIVATE -DALPAKA_LOCAL )
endif()


# Install library
install(TARGETS ${FastCaloGpu_LIB}
  DESTINATION ${CMAKE_INSTALL_LIBDIR}
)





