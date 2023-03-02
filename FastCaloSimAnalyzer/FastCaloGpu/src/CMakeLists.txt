# Copyright (C) 2002-2019 CERN for the benefit of the ATLAS collaboration

# "First-class" CUDA support needs at least CMake 3.10.
cmake_minimum_required( VERSION 3.10 )

# Set up the "modern" usage of CUDA with CMake.
find_package(CUDA REQUIRED)
enable_language( CUDA )


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
	#set(OpenMP_FLAGS "-fopenmp --offload-arch=gfx906") ##lambda3
	##set(OpenMP_FLAGS "-mp=gpu -gpu=cc86 -cuda -Minfo=mp") ## for nvc++
	set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_FLAGS} ${OpenMP_OPT_FLAGS} ${OpenMP_OPT_RMRKS}")
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_FLAGS} ${OpenMP_OPT_FLAGS} ${OpenMP_OPT_RMRKS}")
else()
  message(WARNING "Configuring without OpenMP!")
endif()

#Need to do other way.
#set(CMAKE_CUDA_FLAGS "-arch=sm_37 ")
# Sources
set(FastCaloGpu_Srcs
	#GeoRegion.cxx
  #GeoLoadGpu_omp.cxx
  #Rand4Hits.cu
  gpuQ.cu
  #LoadGpuFuncHist_omp.cxx
  #CaloGpuGeneral_omp.cxx
  OpenMPWrapper.cxx
)

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
set(CUDA_LIBRARIES PUBLIC ${CUDA_LIBRARIES})

# Define and build the library
message(STATUS "Creating library target '${FastCaloGpu_LIB}'")
add_library(${FastCaloGpu_LIB} SHARED ${FastCaloGpu_Srcs}  )

target_include_directories(${FastCaloGpu_LIB} PRIVATE ../FastCaloGpu/ } )

target_link_libraries(${FastCaloGpu_LIB} PUBLIC ${CUDA_curand_LIBRARY})

if(ENABLE_OMPGPU)
  target_compile_definitions(${FastCaloGpu_LIB} PRIVATE -DUSE_OMPGPU )
endif()

# Install library
install(TARGETS ${FastCaloGpu_LIB}
  DESTINATION ${CMAKE_INSTALL_LIBDIR}
)





