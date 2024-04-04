/*
  Copyright (C) 2002-2021 CERN for the benefit of the ATLAS collaboration
*/

#ifndef GPU_GENERAL_STRUCT_H
#define GPU_GENERAL_STRUCT_H

#ifdef USE_ATOMICADD
#  define CELL_ENE_T float
#  define CELL_ENE_FAC 1
#  define CELL_CT_T int
#else
#  ifdef USE_STDPAR
#    define CELL_CT_T std::atomic<int>
#    ifdef _NVHPC_STDPAR_NONE
#      define CELL_ENE_T   float
#      define CELL_ENE_FAC 1
#    else
#      define CELL_ENE_T   std::atomic<unsigned long>
#      define CELL_ENE_FAC 1000000
#    endif
#  else
#    define CELL_ENE_T   float
#    define CELL_ENE_FAC 1
#    define CELL_CT_T    int
#  endif
#endif

#include "FH_structs.h"
typedef struct Cell_E {
  unsigned long cellid;
  float         energy;
} Cell_E;

#endif
