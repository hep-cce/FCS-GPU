/*
  Copyright (C) 2002-2021 CERN for the benefit of the ATLAS collaboration
*/

#ifndef FH_STRUCT_H
#define FH_STRUCT_H

typedef struct FHs {
  uint32_t s_MaxValue{ 0 };
  float *low_edge{ nullptr };
  unsigned int nhist{ 0 };
  unsigned int *h_szs{ nullptr };
  uint32_t **h_contents{ nullptr };
  float **h_borders{ nullptr };
  uint32_t *d_contents1D{ nullptr };
  float *d_borders1D{ nullptr };

} FHs;

typedef struct FH2D {
  int nbinsx{ 0 };
  int nbinsy{ 0 };
  float *h_bordersx{ nullptr };
  float *h_bordersy{ nullptr };
  float *h_contents{ nullptr };
} FH2D;

#endif
