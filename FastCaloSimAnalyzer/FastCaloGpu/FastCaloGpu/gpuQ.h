/*
  Copyright (C) 2002-2021 CERN for the benefit of the ATLAS collaboration
*/

#ifdef USE_HIP
 #include "hip/hip_runtime.h"
 #define DRV_T "hip/driver_types.h"
 #define ERR_T hipError_t
#else
 #define DRV_T "driver_types.h"
 #define ERR_T cudaError_t
#endif

#include DRV_T

#ifndef GPUQ_H
#define GPUQ_H

void gpu_assert(ERR_T code, const char *file, const int line);

#ifndef gpuQ
#define gpuQ(ans)                                                              \
  { gpu_assert((ans), __FILE__, __LINE__); }

#endif
#endif
