# Copyright (C) 2002-2017 CERN for the benefit of the ATLAS collaboration
#
# This file collects ATLAS platform functions adapted for FastCaloSim needs
#

# This function is used by the code to get the "compiler portion"
# of the platform name. E.g. for GCC 4.9.2, return "gcc49". In case
# the compiler and version are not understood, the functions returns
# a false value in its second argument.
#
# Usage: atlas_compiler_id( _cmp _isValid )
#
function( atlas_compiler_id compiler isValid )

  # Translate the compiler ID:
  set( _prefix )
  if( "${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU" )
    set( _prefix "gcc" )
  elseif( "${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang" )
    set( _prefix "clang" )
  elseif( "${CMAKE_CXX_COMPILER_ID}" STREQUAL "Intel" )
    set( _prefix "icc" )
  else()
    set( ${compiler} "unknown" PARENT_SCOPE )
    set( ${isValid} FALSE PARENT_SCOPE )
    return()
  endif()
  
  # Translate the compiler version:
  set( _version )
  if( CMAKE_CXX_COMPILER_VERSION MATCHES "^([0-9]+).([0-9]+).*"
        AND NOT ( ( "${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU" ) AND
           ( "7" VERSION_LESS "${CMAKE_CXX_COMPILER_VERSION}" ) ) )
    set( _version "${CMAKE_MATCH_1}${CMAKE_MATCH_2}" )
  elseif( CMAKE_CXX_COMPILER_VERSION MATCHES "^([0-9]+).*" )
    set( _version "${CMAKE_MATCH_1}" )
  endif()

  # For GCC >=7.0 we only take the first digit of the version number:
  if( "${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU" AND NOT
      "${CMAKE_CXX_COMPILER_VERSION}" VERSION_LESS 7 AND
      CMAKE_CXX_COMPILER_VERSION MATCHES "^([0-9]+).*" )
    set( _version "${CMAKE_MATCH_1}" )
  endif()

  # Set the return variables:
  set( ${compiler} "${_prefix}${_version}" PARENT_SCOPE )
  set( ${isValid} TRUE PARENT_SCOPE )

endfunction( atlas_compiler_id )

# This function is used to get a compact OS designation for the platform
# name. Like "slc6", "mac1010" or "ubuntu1604".
#
# Usage: atlas_os_id( _os _isValid )
#
function( atlas_os_id os isValid )

    # Return cached result if possible:
    if ( ATLAS_OS_ID )
       set( ${os} ${ATLAS_OS_ID} PARENT_SCOPE )
       set( ${isValid} TRUE PARENT_SCOPE )
       return()
    endif()

   set( _name )
   if( APPLE )
      # Set a default version in case the following should not work:
      set( _name "mac1010" )
      # Get the MacOS X version number from the command line:
      execute_process( COMMAND sw_vers -productVersion
         TIMEOUT 30
         OUTPUT_VARIABLE _macVers )
      # Parse the variable, which should be in the form "X.Y.Z", or
      # possibly just "X.Y":
      if( _macVers MATCHES "^([0-9]+).([0-9]+).*" )
         set( _name "mac${CMAKE_MATCH_1}${CMAKE_MATCH_2}" )
      else()
         set( ${os} "unknown" PARENT_SCOPE )
         set( ${isValid} FALSE PARENT_SCOPE )
         return()
      endif()
   elseif( UNIX )
      # Set a default version in case the following should not work:
      set( _name "slc6" )
      # Get the linux release ID:
      execute_process( COMMAND lsb_release -i
         TIMEOUT 30
         OUTPUT_VARIABLE _linuxId )
      # Translate it to a shorthand according to our own naming:
      set( _linuxShort )
      if( _linuxId MATCHES "Scientific" )
         set( _linuxShort "slc" )
      elseif( _linuxId MATCHES "Ubuntu" )
         set( _linuxShort "ubuntu" )
      elseif( _linuxId MATCHES "CentOS" )
         set( _linuxShort "centos" )
      elseif( _linuxId MATCHES "Gentoo" )
         set( _linuxShort "gentoo" )
      elseif( _linuxId MATCHES "Rocky" )
         set( _linuxShort "rocky" )
      else()
         message( WARNING "Linux flavour not recognised: ${_linuxId}" )
         set( _linuxShort "linux" )
      endif()
      # Get the linux version number:
      execute_process( COMMAND lsb_release -r
         TIMEOUT 30
         OUTPUT_VARIABLE _linuxVers )
      # Try to parse it:
      if( _linuxVers MATCHES "^Release:[^0-9]*([0-9]+)\\.([0-9]+).*" )
         if( "${_linuxShort}" STREQUAL "ubuntu" )
            # For Ubuntu include the minor version number as well:
            set( _name "${_linuxShort}${CMAKE_MATCH_1}${CMAKE_MATCH_2}" )
         else()
            # For other Linux flavours use only the major version number:
            set( _name "${_linuxShort}${CMAKE_MATCH_1}" )
         endif()
      elseif(_linuxVers MATCHES "^Release:[^0-9]*([0-9]+).*" )
            set( _name "${_linuxShort}${CMAKE_MATCH_1}" )
      else()
         set( ${os} "unknown" PARENT_SCOPE )
         set( ${isValid} FALSE PARENT_SCOPE )
         return()
      endif()
   else()
      set( ${os} "unknown" PARENT_SCOPE )
      set( ${isValid} FALSE PARENT_SCOPE )
      return()
   endif()

   # Set and cache the return values:
   set( ATLAS_OS_ID "${_name}" CACHE INTERNAL "Compact platform name" )
   set( ${os} ${_name} PARENT_SCOPE )
   set( ${isValid} TRUE PARENT_SCOPE )

endfunction( atlas_os_id )

# This function is used internally to construct a platform name for a
# project. Something like: "x86_64-slc6-gcc48-opt".
#
# Usage: atlas_platform_id( _platform )
#
function( atlas_platform_id platform )

  # Get the OS's name:
  atlas_os_id( _os _valid )
  if( NOT _valid )
    set( ${platform} "generic" PARENT_SCOPE )
    return()
  endif()

  # Get the compiler name:
  atlas_compiler_id( _cmp _valid )
  if( NOT _valid )
    set( ${platform} "generic" PARENT_SCOPE )
    return()
  endif()

  # Construct the postfix of the platform name:
  if( CMAKE_BUILD_TYPE STREQUAL "Debug" )
    set( _postfix "dbg" )
  else()
    set( _postfix "opt" )
  endif()

  # Set the platform return value:
  set( ${platform} "${CMAKE_SYSTEM_PROCESSOR}-${_os}-${_cmp}-${_postfix}"
    PARENT_SCOPE )

endfunction( atlas_platform_id )
