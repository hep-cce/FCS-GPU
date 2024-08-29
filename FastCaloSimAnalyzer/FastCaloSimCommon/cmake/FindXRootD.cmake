# Copyright (C) 2002-2019 CERN for the benefit of the ATLAS collaboration
#
# - Locate Xrootd library
# Defines:
#
#  XROOTD_FOUND
#  XROOTD_INCLUDE_DIR
#  XROOTD_INCLUDE_DIRS
#  XROOTD_<component>_LIBRARY
#  XROOTD_<component>_FOUND
#  XROOTD_LIBRARIES
#

set(XROOTD_LIBRARIES)
set(XROOTD_INCLUDE_DIRS)

# Find the xrootd executable, and set up the binary path using it:
find_program(XROOTD_EXECUTABLE xrootd
  PATH_SUFFIXES bin
  PATHS $ENV{XROOTD__HOME}
  $ENV{XROOTD_HOME}
  /usr )

get_filename_component(XROOTD_BINARY_PATH ${XROOTD_EXECUTABLE} PATH)
get_filename_component(XROOTD_HOME "${XROOTD_BINARY_PATH}/.." ABSOLUTE)

# Find the xrootd client library
find_library(XROOTD_XrdCl_LIBRARY
  NAMES XrdCl
  PATH_SUFFIXES lib lib64
  PATHS ${XROOTD_HOME})

list(APPEND XROOTD_LIBRARIES
  ${XROOTD_XrdCl_LIBRARY})
set(XROOTD_XrdCl_FOUND TRUE)

# Find the include directory
find_path(XROOTD_INCLUDE_DIR XrdVersion.hh
  PATH_SUFFIXES include/xrootd
  PATHS ${XROOTD_HOME})

list(APPEND XROOTD_INCLUDE_DIRS
  ${XROOTD_INCLUDE_DIR})

# Handle the standard find_package arguments:
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(XRootD
  DEFAULT_MSG
  XROOTD_INCLUDE_DIR
  XROOTD_LIBRARIES)

mark_as_advanced(XROOTD_FOUND XROOTD_HOME XROOTD_INCLUDE_DIRS
 XROOTD_LIBRARIES XROOTD_EXECUTABLE XROOTD_BINARY_PATH)
