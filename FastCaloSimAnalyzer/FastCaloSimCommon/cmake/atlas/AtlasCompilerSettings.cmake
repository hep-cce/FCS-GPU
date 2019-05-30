# Copyright (C) 2002-2018 CERN for the benefit of the ATLAS collaboration
#
# This file collects settings fine-tuning all the compiler and linker options
# used in an ATLAS build in one place. Additional changes are made for FastCaloSim
#

# Guard this file against multiple inclusions:
get_property( _compilersSet GLOBAL PROPERTY ATLAS_COMPILERS_SET SET )
if( _compilersSet )
   unset( _compilersSet )
   return()
endif()
set_property( GLOBAL PROPERTY ATLAS_COMPILERS_SET TRUE )

# Tell the user what we're doing:
message( STATUS "Setting ATLAS specific build flags" )

# Include the necessary module(s):
include( CheckTypeSize )
include( CheckIncludeFileCXX )
include( CheckFunctionExists )
include( CheckSymbolExists )

# Set C++17 for GCC 7 and above
# Can remove this logic when we stop supporting GCC 6
if( CMAKE_COMPILER_IS_GNUCXX AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 7)
  set( CMAKE_CXX_STANDARD 14 CACHE STRING "C++ standard used for the build" )
else()
  set( CMAKE_CXX_STANDARD 17 CACHE STRING "C++ standard used for the build" )
endif()

set( CMAKE_CXX_EXTENSIONS FALSE CACHE BOOL "(Dis)allow using GNU extensions" )


# Set the definitions needed everywhere:
add_definitions( -DHAVE_PRETTY_FUNCTION )

# Function helping with setting up variable type checks
function( _check_for_type typeName definition )
   check_type_size( ${typeName} ${definition} LANGUAGE CXX )
   if( HAVE_${definition} )
      add_definitions( -DHAVE_${definition} )
   endif()
endfunction( _check_for_type )

# Function helping with setting up header checks
function( _check_for_include include definition )
   check_include_file_cxx( ${include} ${definition} )
   if( ${definition} )
      add_definitions( -D${definition} )
   endif()
endfunction( _check_for_include )

# Clean up:
unset( _check_for_type )
unset( _check_for_include )

# We work only in 64-bit mode right now:
add_definitions( -DHAVE_64_BITS -D__IDENTIFIER_64BIT__ )

# Flag showing that we are building ATLAS code:
add_definitions( -DATLAS )


# Set default values for the optimisation settings, ones that could
# be overridden from the command line.
if( CMAKE_COMPILER_IS_GNUCXX )

   # Provide custom defaults for GCC. The only modification wrt. the
   # CMake default values, at the time of writing, is to use -O2 for release
   # mode instead of -O3.
   set( ATLAS_CXX_FLAGS_RELEASE "-DNDEBUG -O2"
      CACHE STRING "Default optimisation settings for Release mode" )
   set( ATLAS_CXX_FLAGS_RELWITHDEBINFO "-DNDEBUG -O2 -g"
      CACHE STRING "Default optimisation settings for RelWithDebInfo mode" )
   set( ATLAS_CXX_FLAGS_DEBUG "-g"
      CACHE STRING "Default optimisation settings for Debug mode" )

else()

   # For all other compilers just accept the CMake defaults.
   set( ATLAS_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE}"
      CACHE STRING "Default optimisation settings for Release mode" )
   set( ATLAS_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO}"
      CACHE STRING "Default optimisation settings for RelWithDebInfo mode" )
   set( ATLAS_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG}"
      CACHE STRING "Default optimisation settings for Debug mode" )

endif()

# Override the default values forcefully. Since by this time the cache
# values have already been set by CMake.
set( CMAKE_CXX_FLAGS_RELEASE "${ATLAS_CXX_FLAGS_RELEASE}"
   CACHE STRING "Default optimisation settings for Release mode" FORCE )
set( CMAKE_CXX_FLAGS_RELWITHDEBINFO "${ATLAS_CXX_FLAGS_RELWITHDEBINFO}"
   CACHE STRING "Default optimisation settings for RelWithDebInfo mode" FORCE )
set( CMAKE_CXX_FLAGS_DEBUG "${ATLAS_CXX_FLAGS_DEBUG}"
   CACHE STRING "Default optimisation settings for Release mode" FORCE )

# Function helping with extending global flags
function( _add_flag name value )

   # Escape special characters in the value:
   set( matchedValue "${value}" )
   foreach( c "*" "." "^" "$" "+" "?" )
      string( REPLACE "${c}" "\\${c}" matchedValue "${matchedValue}" )
   endforeach()

   # Check if the variable already has this value in it:
   if( "${${name}}" MATCHES "${matchedValue}" )
      return()
   endif()

   # If not, then let's add it now:
   set( ${name} "${${name}} ${value}" CACHE STRING
      "Compiler setting" FORCE )

endfunction( _add_flag )

# Fortran settings:
_add_flag( CMAKE_Fortran_FLAGS "-ffixed-line-length-132" )
_add_flag( CMAKE_Fortran_FLAGS "-DFVOIDP=INTEGER*8" )
_add_flag( CMAKE_Fortran_FLAGS_RELEASE "-funroll-all-loops" )
_add_flag( CMAKE_Fortran_FLAGS_RELWITHDEBINFO "-funroll-all-loops" )

# Turn on compiler warnings for the relevant build modes. In a default
# build don't use extra warnings.
if( CMAKE_COMPILER_IS_GNUCXX )
   foreach( mode RELEASE RELWITHDEBINFO DEBUG )
      _add_flag( CMAKE_CXX_FLAGS_${mode} "-Wall" )
      _add_flag( CMAKE_CXX_FLAGS_${mode} "-Wno-long-long" )
      _add_flag( CMAKE_CXX_FLAGS_${mode} "-Wno-deprecated" )
      _add_flag( CMAKE_CXX_FLAGS_${mode} "-Wno-unused-local-typedefs" )
      _add_flag( CMAKE_CXX_FLAGS_${mode} "-Wwrite-strings" )
      _add_flag( CMAKE_CXX_FLAGS_${mode} "-Wpointer-arith" )
      _add_flag( CMAKE_CXX_FLAGS_${mode} "-Woverloaded-virtual" )
      _add_flag( CMAKE_CXX_FLAGS_${mode} "-Wextra" )
      _add_flag( CMAKE_CXX_FLAGS_${mode} "-Werror=return-type" )
   endforeach()
   _add_flag( CMAKE_CXX_FLAGS_DEBUG "-fsanitize=undefined" )
endif()
#foreach( mode RELEASE RELWITHDEBINFO DEBUG )
#   _add_flag( CMAKE_CXX_FLAGS_${mode} "-pedantic" )
#endforeach()

#
# Set up the linker flags:
#

# By default use the --as-needed option for everything. Remember that this
# behaviour can be turned off in individual packages using
# atlas_disable_as_needed().
if( NOT APPLE )
   _add_flag( CMAKE_EXE_LINKER_FLAGS    "-Wl,--as-needed" )
   _add_flag( CMAKE_SHARED_LINKER_FLAGS "-Wl,--as-needed" )
   _add_flag( CMAKE_MODULE_LINKER_FLAGS "-Wl,--as-needed" )

   _add_flag( CMAKE_SHARED_LINKER_FLAGS "-Wl,--no-undefined" )
   _add_flag( CMAKE_MODULE_LINKER_FLAGS "-Wl,--no-undefined" )
endif()

# In 64-bit build mode this overrides the default library segment alignment
# of 1 MB:
if( "${CMAKE_SIZEOF_VOID_P}" EQUAL "8" AND NOT APPLE )
   _add_flag( CMAKE_SHARED_LINKER_FLAGS "-Wl,-z,max-page-size=0x1000" )
   _add_flag( CMAKE_MODULE_LINKER_FLAGS "-Wl,-z,max-page-size=0x1000" )
endif()

# This has some benefits in the speed of loading libraries:
if( NOT APPLE )
   _add_flag( CMAKE_EXE_LINKER_FLAGS    "-Wl,--hash-style=both" )
   _add_flag( CMAKE_SHARED_LINKER_FLAGS "-Wl,--hash-style=both" )
   _add_flag( CMAKE_MODULE_LINKER_FLAGS "-Wl,--hash-style=both" )
endif()

# Clean up:
unset( _add_flag )
