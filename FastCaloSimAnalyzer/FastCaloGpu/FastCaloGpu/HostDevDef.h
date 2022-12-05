#ifndef HOSTDEVDEF_H
#define HOSTDEVDEF_H

#if defined(USE_KOKKOS)
#include <Kokkos_Core.hpp>
#include <Kokkos_Random.hpp>
#define __DEVICE__ KOKKOS_INLINE_FUNCTION
#define __HOST__
#define __HOSTDEV__ KOKKOS_INLINE_FUNCTION
#define __INLINE__
#elif defined(USE_STDPAR)
#define __DEVICE__ __device__
#define __HOST__
#define __HOSTDEV__
#define __INLINE__ inline
#elif defined(__CUDACC__)
#define __DEVICE__ __device__
#define __HOST__ __host__
#define __HOSTDEV__ __host__ __device__
#define __INLINE__ inline
#elif defined(__GNUC__)
#define __DEVICE__
#define __HOST__
#define __HOSTDEV__
#define __INLINE__ inline
#else
#define __DEVICE__ ERROR
#define __HOST__ ERROR
#define __HOSTDEV__ ERROR
#define __INLINE__ ERROR
#endif

#endif
