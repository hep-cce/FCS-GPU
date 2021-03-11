#ifndef GPUQ_H
#define GPUQ_H

#include "hip/hip_runtime.h"
#include <iostream>

#define gpuQ( ans )                                                                                                    \
  { gpu_assert( ( ans ), __FILE__, __LINE__ ); }
void gpu_assert( hipError_t code, const char* file, const int line ) {
  if ( code != hipSuccess ) {
    std::cerr << "gpu_assert: " << hipGetErrorString( code ) << " " << file << " " << line << std::endl;
    exit( code );
  }
}

#endif
