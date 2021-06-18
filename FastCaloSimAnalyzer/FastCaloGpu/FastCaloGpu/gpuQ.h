/*
  Copyright (C) 2002-2021 CERN for the benefit of the ATLAS collaboration
*/

#ifndef GPUQ_H
#define GPUQ_H

#define gpuQ(ans) { gpu_assert((ans), __FILE__, __LINE__); }
void gpu_assert(cudaError_t code, const char *file, const int line)
{
    if (code != cudaSuccess)
    {
        std::cerr << "gpu_assert: " << cudaGetErrorString(code) << " " 
            << file << " " << line << std::endl;
        exit(code);
    }
}

#endif
