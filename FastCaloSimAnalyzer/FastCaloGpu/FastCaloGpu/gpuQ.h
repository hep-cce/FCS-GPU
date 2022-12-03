/*
  Copyright (C) 2002-2023 CERN for the benefit of the ATLAS collaboration
*/

#ifndef GPUQ_H
#define GPUQ_H

#define HIPCALL(x)  if((x)!=0) { \
    printf("hip error at %s:%d\n",__FILE__,__LINE__);\
    exit(EXIT_FAILURE) ; }

#define gpuQ(ans) { gpu_assert((ans), __FILE__, __LINE__); }
void gpu_assert(hipError_t code, const char *file, const int line)
{
    if (code != hipSuccess)
    {
        std::cerr << "gpu_assert: " << hipGetErrorString(code) << " " 
            << file << " " << line << std::endl;
        exit(code);
    }
}

#endif
