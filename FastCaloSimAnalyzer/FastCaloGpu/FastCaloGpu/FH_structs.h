/*
  Copyright (C) 2002-2021 CERN for the benefit of the ATLAS collaboration
*/

#ifndef FH_STRUCT_H
#define FH_STRUCT_H

#include <cstdint>

typedef struct FHs {
  uint32_t      s_MaxValue {0};
  float*        low_edge{0};
  unsigned int  nhist{0};
  unsigned int  mxsz{0};
  unsigned int* h_szs{0};
  uint32_t**    h_contents{0};
  float**       h_borders{0};
  uint32_t*     d_contents1D{0};
  float*        d_borders1D{0};
} FHs;

typedef struct FH2D {
  int    nbinsx{0};
  int    nbinsy{0};
  float* h_bordersx{0};
  float* h_bordersy{0};
  float* h_contents{0};
} FH2D;

#endif
