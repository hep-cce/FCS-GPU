/*
  Copyright (C) 2002-2021 CERN for the benefit of the ATLAS collaboration
*/

#include "gpuQ.h"
#include <iostream>

void gpu_assert(hipError_t code, const char *file, const int line) {
  if (code != hipSuccess) {
    std::cerr << "gpu_assert: " << hipGetErrorString(code) << " " << file
              << " " << line << std::endl;
    exit(code);
  }
}
