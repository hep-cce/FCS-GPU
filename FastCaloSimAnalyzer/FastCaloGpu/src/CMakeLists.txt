# Copyright (C) 2002-2018 CERN for the benefit of the ATLAS collaboration

find_package(CUDA)
#Need to do other way.
#set(CMAKE_CUDA_FLAGS "-arch=sm_37 ")
# Sources
set(FastCaloGpu_Srcs
  GeoLoadGpu.cu
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
cuda_add_library(${FastCaloGpu_LIB} SHARED ${FastCaloGpu_Srcs}  )

target_include_directories(${FastCaloGpu_LIB} PRIVATE ../FastCaloGpu/ } )

target_link_libraries(${FastCaloGpu_LIB} PRIVATE ${CUDA_curand_LIBRARY}) 

# Install library
install(TARGETS ${FastCaloGpu_LIB}
  DESTINATION ${CMAKE_INSTALL_LIBDIR}
)





