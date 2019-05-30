# Copyright (C) 2002-2018 CERN for the benefit of the ATLAS collaboration
#

# Helper macro setting up installation targets. Not for use outside of this
# file.
#
macro( _atlas_create_install_target tgtName )

   if( NOT TARGET ${tgtName} )
      add_custom_target( ${tgtName} ALL SOURCES
         $<TARGET_PROPERTY:${tgtName},INSTALLED_FILES> )
   endif()

endmacro( _atlas_create_install_target )

# This is a generic function for installing practically any type of file
# from a package into both the build and the install areas. Behind the scenes
# it is used by most of the functions of this file.
#
# Based on atlas_install_generic.
#
# Usage: fcs_install_generic( dir/file1 dir/dir2...
#                             DESTINATION dir
#                             [BUILD_DESTINATION dir]
#                             [TYPENAME type]
#                             [EXECUTABLE] )
#
function( fcs_install_generic )

   # Parse the options given to the function:
   cmake_parse_arguments( ARG "EXECUTABLE" "TYPENAME;DESTINATION;BUILD_DESTINATION" "" ${ARGN} )

   # If there are no file/directory names given to the function, return now:
   if( NOT ARG_UNPARSED_ARGUMENTS )
      message( WARNING "Function received no file/directory arguments" )
      return()
   endif()
   if( NOT ARG_DESTINATION )
      message( WARNING "No destination was specified" )
      return()
   endif()

   # Create an installation target for this type:
   if( ARG_TYPENAME )
       set( _tgtName FCSInstall_${ARG_TYPENAME} )
    else()
       set( _tgtName FCSInstall_Generic )
    endif()

   _atlas_create_install_target( ${_tgtName} )

   # Expand possible wildcards:
   file( GLOB_RECURSE _files RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
      ${ARG_UNPARSED_ARGUMENTS} )

   # Define what the build and install area destinations should be:
   set( _buildDest ${CMAKE_BINARY_DIR}/${ATLAS_PLATFORM}/${ARG_DESTINATION} )
   set( _installDest ${ARG_DESTINATION} )

   # Now loop over all file names:
   foreach( _file ${_files} )
      # Set up its installation into the build area:
      file( RELATIVE_PATH _target
         ${_buildDest} ${CMAKE_CURRENT_SOURCE_DIR}/${_file} )
      get_filename_component( _filename ${_file} NAME )
      add_custom_command( OUTPUT ${_buildDest}/${_filename}
         COMMAND ${CMAKE_COMMAND} -E make_directory ${_buildDest}
         COMMAND ${CMAKE_COMMAND} -E create_symlink ${_target}
         ${_buildDest}/${_filename} )
      set_property( DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} APPEND PROPERTY
         ADDITIONAL_MAKE_CLEAN_FILES
         ${_buildDest}/${_filename} )
      # Add it to the installation target:
      set_property( TARGET ${_tgtName} APPEND PROPERTY
         INSTALLED_FILES ${_buildDest}/${_filename} )
      # Set up its installation into the install area:
      if( IS_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/${_file} )
         install( DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/${_file}
            DESTINATION ${_installDest}
            USE_SOURCE_PERMISSIONS
            PATTERN ".svn" EXCLUDE )
      else()
         # In case this turns out to be a symbolic link, install the actual
         # file that it points to, using the name of the symlink.
         get_filename_component( _realpath
            ${CMAKE_CURRENT_SOURCE_DIR}/${_file} REALPATH )
         if( ARG_EXECUTABLE )
            install( PROGRAMS ${_realpath}
               DESTINATION ${_installDest}
               RENAME ${_filename} )
         else()
            install( FILES ${_realpath}
               DESTINATION ${_installDest}
               RENAME ${_filename} )
         endif()
      endif()
   endforeach()

endfunction( fcs_install_generic )

# This function installs files from the package into the
# right place in both the build and the install directories.
#
# Based on atlas_install_python_modules.
#
# Usage: fcs_install_files( someFiles )
#
function( fcs_install_files )

   cmake_parse_arguments( ARG "" "" "" ${ARGN} )

   # Call the generic function:
   fcs_install_generic( ${ARG_UNPARSED_ARGUMENTS}
      DESTINATION "."
      BUILD_DESTINATION ${CMAKE_OUTPUT_DIRECTORY}
      TYPENAME File )

endfunction( fcs_install_files )

# This function installs headers from the package into the
# right place in both the build and the install directories.
#
# Based on atlas_install_python_modules.
#
# Usage: fcs_install_headers( *.h
#                             PACKAGE pkgName )
#
function( fcs_install_headers )

   cmake_parse_arguments( ARG "" "" "" ${ARGN} )

   # Call the generic function:
   fcs_install_generic( ${ARG_UNPARSED_ARGUMENTS}
      DESTINATION ${CMAKE_INSTALL_INCDIR}
      BUILD_DESTINATION ${CMAKE_INCLUDE_OUTPUT_DIRECTORY}
      TYPENAME Headers )

endfunction( fcs_install_headers )

# This function installs python modules from the package into the
# right place in both the build and the install directories.
#
# Based on atlas_install_python_modules.
#
# Usage: fcs_install_python_module( python/SomeDir/*.py
#                                  PACKAGE pkgName )
#
function( fcs_install_python_module )
    
   cmake_parse_arguments( ARG "" "PACKAGE" "" ${ARGN} )
   if( NOT ARG_PACKAGE )
      message( WARNING "No package name was specified" )
      return()
   endif()

   # Call the generic function:
   fcs_install_generic( ${ARG_UNPARSED_ARGUMENTS}
      DESTINATION ${CMAKE_INSTALL_PYTHONDIR}/${ARG_PACKAGE}
      BUILD_DESTINATION ${CMAKE_PYTHON_OUTPUT_DIRECTORY}/${ARG_PACKAGE}
      TYPENAME Python )

endfunction( fcs_install_python_module )

# This function installs binaries from the package into the
# right place in both the build and the install directories.
#
# Based on atlas_install_python_modules.
#
# Usage: fcs_install_binary( bin/someBin )
#
function( fcs_install_binary )
    
   cmake_parse_arguments( ARG "" "" "" ${ARGN} )

   # Call the generic function:
   fcs_install_generic( ${ARG_UNPARSED_ARGUMENTS}
      DESTINATION ${CMAKE_INSTALL_BINDIR}
      BUILD_DESTINATION ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}
      TYPENAME Binary
      EXECUTABLE )

endfunction( fcs_install_binary )

# This function installs data from the package into the
# right place in both the build and the install directories.
#
# Based on atlas_install_python_modules.
#
# Usage: fcs_install_data( someFile )
#
function( fcs_install_data )

   cmake_parse_arguments( ARG "" "" "" ${ARGN} )

   # Call the generic function:
   fcs_install_generic( ${ARG_UNPARSED_ARGUMENTS}
      DESTINATION ${CMAKE_INSTALL_DATADIR}
      BUILD_DESTINATION ${CMAKE_DATA_OUTPUT_DIRECTORY}
      TYPENAME Data )

endfunction( fcs_install_data )
