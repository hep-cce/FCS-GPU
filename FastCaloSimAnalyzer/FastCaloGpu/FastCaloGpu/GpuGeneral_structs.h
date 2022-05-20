/*
  Copyright (C) 2002-2021 CERN for the benefit of the ATLAS collaboration
*/

#ifndef GPU_GENERAL_STRUCT_H
#define GPU_GENERAL_STRUCT_H

#ifdef USE_ATOMICADD
# define CELL_ENE_T float
# define CELL_ENE_FAC 1
# define CELL_CT_T int
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

typedef struct HitParams {
  int    index; // simulate index for gpu simulation
  int    cs;
  float  E;
  int    pdgId;
  double charge;
  bool   is_phi_symmetric;
  long   nhits;
  FHs*   f1d;
  FH2D*  f2d;
  double extrapol_eta_ent;
  double extrapol_eta_ext;
  double extrapol_phi_ent;
  double extrapol_phi_ext;
  double extrapol_r_ent;
  double extrapol_r_ext;
  double extrapol_z_ent;
  double extrapol_z_ext;
  float  extrapWeight;
  bool   cmw; // Do CellMapingWiggle or direct CellMapping
} HitParams;
#endif
