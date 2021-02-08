# Copyright (C) 2002-2018 CERN for the benefit of the ATLAS collaboration

# Discover ATLAS platform
atlas_platform_id(ATLAS_PLATFORM)

# Setup local Athena path
set(ATHENA_PATH "${CMAKE_SOURCE_DIR}/../athena" CACHE STRING "Local Athena path")
message(STATUS "Using Athena from '${ATHENA_PATH}'")

# Setup install directories
set(CMAKE_INSTALL_BINDIR "bin"
    CACHE STRING "Executable installation directory" FORCE)
set(CMAKE_INSTALL_LIBDIR "lib"
    CACHE STRING "Library installation directory" FORCE)
set(CMAKE_INSTALL_INCDIR "include"
    CACHE STRING "Header installation directory" FORCE)
set(CMAKE_INSTALL_PYTHONDIR "python"
    CACHE STRING "Python installation directory" FORCE)
set(CMAKE_INSTALL_DATADIR "data"
    CACHE STRING "Data installation directory" FORCE)

# Setup output directories
set(CMAKE_OUTPUT_DIRECTORY
    "${CMAKE_BINARY_DIR}/${ATLAS_PLATFORM}" CACHE STRING
    "Directory used to store files during compilation" FORCE)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY
    "${CMAKE_BINARY_DIR}/${ATLAS_PLATFORM}/bin" CACHE STRING
    "Directory used to store executables during compilation" FORCE)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY
    "${CMAKE_BINARY_DIR}/${ATLAS_PLATFORM}/lib" CACHE STRING
    "Directory used to store shared libraries during compilation" FORCE)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY
    "${CMAKE_BINARY_DIR}/${ATLAS_PLATFORM}/lib" CACHE STRING
    "Directory used to store static libraries during compilation" FORCE)
set(CMAKE_INCLUDE_OUTPUT_DIRECTORY
    "${CMAKE_BINARY_DIR}/${ATLAS_PLATFORM}/include" CACHE STRING
    "Directory used to look up header files during compilation" FORCE)
set(CMAKE_PYTHON_OUTPUT_DIRECTORY
    "${CMAKE_BINARY_DIR}/${ATLAS_PLATFORM}/python" CACHE STRING
    "Directory collecting python modules in the build area" FORCE)
set(CMAKE_DATA_OUTPUT_DIRECTORY
    "${CMAKE_BINARY_DIR}/${ATLAS_PLATFORM}/data" CACHE STRING
    "Directory collecting data in the build area" FORCE)
