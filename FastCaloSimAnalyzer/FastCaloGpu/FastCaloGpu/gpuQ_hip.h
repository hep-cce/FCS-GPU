/*
  Copyright (C) 2002-2021 CERN for the benefit of the ATLAS collaboration
*/

#include "hip/driver_types.h"

#ifndef GPUQ_H
#define GPUQ_H

void gpu_assert(hipError_t code, const char *file, const int line);

#ifndef gpuQ
#define gpuQ(ans)                                                              \
  { gpu_assert((ans), __FILE__, __LINE__); }

#endif
#endif
