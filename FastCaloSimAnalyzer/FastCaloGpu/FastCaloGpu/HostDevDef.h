#ifndef HOSTDEVDEF_H
#define HOSTDEVDEF_H

#if defined (USE_KOKKOS)
#  include <Kokkos_Core.hpp>
#  include <Kokkos_Random.hpp>
#  define __DEVICE__  KOKKOS_INLINE_FUNCTION
#  define __HOST__
#  define __HOSTDEV__ KOKKOS_INLINE_FUNCTION
#elif defined (USE_STDPAR)
#  if defined (_NVHPC_STDPAR_NONE)
#    define __DEVICE__
#  else
#    define __DEVICE__ __device__
#  endif
#  define __HOST__
#  define __HOSTDEV__
#elif defined (__CUDACC__)
#  define __DEVICE__ __device__
#  define __HOST__   __host__
#  define __HOSTDEV__ __host__ __device__
#elif defined (__GNUC__)
#  define __DEVICE__
#  define __HOST__
#  define __HOSTDEV__
#elif defined (USE_ALPAKA)
#  define __DEVICE__ ALPAKA_FN_ACC
#  define __HOST__   ALPAKA_FN_HOST
#  define __HOSTDEV__ ALPAKA_FN_HOST_ACC
#else
#  define __DEVICE__  ERROR
#  define __HOST__    ERROR
#  define __HOSTDEV__ ERROR
#endif

#endif
