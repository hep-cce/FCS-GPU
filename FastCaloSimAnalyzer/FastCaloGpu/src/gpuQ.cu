#include "gpuQ.h"
#include <iostream>

void gpu_assert( cudaError_t code, const char* file, const int line ) {
  if ( code != cudaSuccess ) {
    std::cerr << "gpu_assert: " << cudaGetErrorString( code ) << " " << file << " " << line << std::endl;
    exit( code );
  }
}
