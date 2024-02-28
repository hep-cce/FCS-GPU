# Copyright (C) 2002-2018 CERN for the benefit of the ATLAS collaboration

# Add FCS standalone library
#
# Usage: fcs_add_library( arguments... )
#
function(fcs_add_library)
  message(STATUS "Creating library target '${ARGV0}'")
  add_library(${ARGV})
  if(USE_KOKKOS) 
    target_link_libraries(${ARGV0} Kokkos::kokkos) 
  endif() 
  target_compile_definitions(${ARGV0} PRIVATE ${FCS_CommonDefinitions})
  target_include_directories(${ARGV0} PRIVATE . ${CMAKE_SOURCE_DIR})
endfunction()

# Add FCS standalone dependency
#
# Usage: fcs_add_dependency( target dependency )
#
function(fcs_add_dependency target dependency)
  message(STATUS "Adding dependency '${dependency}' to target '${target}'")
  add_dependencies(${target} ${dependency})
  target_link_libraries(${target} ${dependency})
  target_include_directories(${target} PRIVATE ${${dependency}_Includes})
endfunction()


# Add ROOT dictionary dependency
#
# Usage: fcs_dictionary_dependency( dependency )
#
function(fcs_dictionary_dependency dependency)
  # Global include is needed for dictionary generation to work
  include_directories(${${dependency}_Includes})
endfunction()


# Define and build FCS task
#
# Usage: fcs_make_task( task_name
#                       SOURCE main_src
#                       DEPENDENCY dependencies
#                     )
#
function(fcs_make_task)
  # Parse the options given to the function:
  cmake_parse_arguments(ARG "" "" "SOURCE;DEPENDENCY" ${ARGN})

  # If there are no file/directory names given to the function, return now:
  if(NOT ARG_SOURCE)
    message(WARNING "Function received no sources arguments")
    return()
  endif()
  if(NOT ARG_DEPENDENCY)
    message(WARNING "No dependencies have been defined")
    return()
  endif()

  set(_target ${ARG_UNPARSED_ARGUMENTS})

  message(STATUS "Creating executable target '${_target}'")

  add_executable(${_target} ${ARG_SOURCE})
  target_compile_definitions(${_target} PRIVATE ${FCS_CommonDefinitions})
  target_include_directories(${_target} PRIVATE ${PROJECT_SRC_DIR})

  if(USE_STDPAR)
    target_compile_options(${_target} PRIVATE $<$<COMPILE_LANG_AND_ID:CXX,GNU>:
      ${STDPAR_DIRECTIVE}> )
    target_link_options(${_target} PRIVATE ${STDPAR_DIRECTIVE})
  elseif(USE_KOKKOS)
    target_link_libraries(${_target} Kokkos::kokkos) 
  endif()

  
  foreach(_dependency ${ARG_DEPENDENCY})
    fcs_add_dependency(${_target} ${_dependency})
  endforeach()

  install(TARGETS ${_target}
    DESTINATION ${CMAKE_INSTALL_BINDIR}
  )
endfunction()
