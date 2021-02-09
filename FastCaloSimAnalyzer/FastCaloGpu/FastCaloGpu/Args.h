/*
  Copyright (C) 2002-2021 CERN for the benefit of the ATLAS collaboration
*/

#ifndef GPUARGS_H
#define GPUARGS_H

#include "FH_structs.h"
#include "GpuGeneral_structs.h"

#define MAXHITS 200000
#define MAXBINS 1024
#define MAXHITCT 2000

class GeoGpu;

typedef struct Chain0_Args {

  bool debug;

  double extrapol_eta_ent;
  double extrapol_phi_ent;
  double extrapol_r_ent;
  double extrapol_z_ent;
  double extrapol_eta_ext;
  double extrapol_phi_ext;
  double extrapol_r_ext;
  double extrapol_z_ext;

  float extrapWeight;
  float E;

  int    pdgId;
  double charge;
  int    cs;
  bool   is_phi_symmetric;
  float* rand;
  int    nhits;
  void*  rd4h;

  FH2D* fh2d;
  FHs*  fhs;
  FH2D  fh2d_h; // host struct so we have info
  FHs   fhs_h;  // host struct

  GeoGpu* geo;

  bool is_first; // first event
  bool is_last;  // last event

  // bool * hitcells_b ;  // GPU array of whether a cell got hit
  // unsigned long * hitcells ;//GPU pointer for hit cell index for each hit
  // unsigned long * hitcells_l ; // GPU pointer for uniq  hitcell indexes
  int*          hitcells_ct; // GPU pointer for number of uniq hit cells
  unsigned long ncells;
  unsigned int  maxhitct;

  float*  cells_energy; // big, all cells, ~ 200K array
  Cell_E* hitcells_E;   // array with only hit cells (Mem maxhitct )
  Cell_E* hitcells_E_h; // host array

  unsigned int* hitcounts_b; // GPU pointer for interm blockwise result of hit counts

  // unsigned long * hitcells_h ; //Host array of hit cell index
  // int * hitcells_ct_h ; // host array of corresponding hit cell counts
  unsigned int ct; // cells got hit for the event

} Chain0_Args;

#endif
