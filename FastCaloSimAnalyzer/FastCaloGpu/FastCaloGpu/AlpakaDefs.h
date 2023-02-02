/**
 **/

#ifndef ALPAKADEFS_H
#define ALPAKADEFS_H

#include <alpaka/alpaka.hpp>
#include "GeoGpu_structs.h"
#include "GpuGeneral_structs.h"
#include "FH_structs.h"

using Dim = alpaka::DimInt<1>;
using Idx = std::size_t;
using Vec = alpaka::Vec<Dim, Idx>;

// Default accelerator used by the application
namespace alpaka {
  template<class TDim, class TIdx>
#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED)
    using FccDefaultAcc = alpaka::AccGpuCudaRt<TDim, TIdx>;
#elif defined(ALPAKA_ACC_GPU_HIP_ENABLED)
    using FccDefaultAcc = alpaka::AccGpuHipRt<TDim, TIdx>;
#elif defined(ALPAKA_ACC_CPU_B_TBB_T_SEQ_ENABLED)
    using FccDefaultAcc = alpaka::AccCpuTbbBlocks<TDim, TIdx>;
#elif defined(ALPAKA_ACC_CPU_B_SEQ_T_THREADS_ENABLED)
    using FccDefaultAcc = alpaka::AccCpuThreads<TDim, TIdx>;
#elif defined(ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED)
    using FccDefaultAcc = alpaka::AccCpuSerial<TDim, TIdx>;
#else
    class FccDefaultAcc;
#   warning "No supported backend selected for default Alpaka accelerator"
#endif
}

using Acc = alpaka::FccDefaultAcc<Dim, Idx>;

using Host = alpaka::DevCpu;
using QueueProperty = alpaka::Blocking;
using QueueAcc = alpaka::Queue<Acc, QueueProperty>;
using WorkDiv = alpaka::WorkDivMembers<Dim, Idx>;

using BufHost = alpaka::Buf<Host, float, Dim, Idx>;
using BufAcc = alpaka::Buf<Acc, float, Dim, Idx>;

using BufHostCaloDDE = alpaka::Buf<Host, CaloDetDescrElement, Dim, Idx>;
using BufHostSampleIndex = alpaka::Buf<Host, Rg_Sample_Index, Dim, Idx>;
using BufHostGeoRegion  = alpaka::Buf<Host, GeoRegion, Dim, Idx>;
using BufHostGeoGpu = alpaka::Buf<Host, GeoGpu, Dim, Idx>;
using BufHostLongLong = alpaka::Buf<Host, long long, Dim, Idx>;
using BufHostLong = alpaka::Buf<Host, long, Dim, Idx>;
using BufHostUnsigned = alpaka::Buf<Host, unsigned, Dim, Idx>;
using BufHostUint32 = alpaka::Buf<Host, uint32_t, Dim, Idx>;

using BufAccCaloDDE = alpaka::Buf<Acc, CaloDetDescrElement, Dim, Idx>;
using BufAccSampleIndex = alpaka::Buf<Acc, Rg_Sample_Index, Dim, Idx>;
using BufAccGeoRegion  = alpaka::Buf<Acc, GeoRegion, Dim, Idx>;
using BufAccGeoGpu = alpaka::Buf<Acc, GeoGpu, Dim, Idx>;
using BufAccLongLong = alpaka::Buf<Acc, long long, Dim, Idx>;
using BufAccLong = alpaka::Buf<Acc, long, Dim, Idx>;
using BufAccUnsigned = alpaka::Buf<Acc, unsigned, Dim, Idx>;
using BufAccUint32 = alpaka::Buf<Acc, uint32_t, Dim, Idx>;
using BufAccHitParams = alpaka::Buf<Acc, HitParams, Dim, Idx>;

// This engine was chosen only because it is used by one of
// the Alpaka examples
template<typename TAcc>
using RandomEngine = alpaka::rand::Philox4x32x10<TAcc>;

using BufHostEngine = alpaka::Buf<Host, RandomEngine<Acc>, Dim, Idx>;
using BufAccEngine = alpaka::Buf<Acc, RandomEngine<Acc>, Dim, Idx>;

// The choice of NUM_STATES value is totally random
unsigned constexpr NUM_STATES = 1000;

using CellsEnergy = alpaka::Buf<Acc,CELL_ENE_T, Dim, Idx>;
using CellE = alpaka::Buf<Acc,Cell_E, Dim, Idx>;
using CellEHost = alpaka::Buf<Host,Cell_E, Dim, Idx>;
using CellCtT = alpaka::Buf<Acc, CELL_CT_T, Dim, Idx>;
using CellCtTHost = alpaka::Buf<Host, CELL_CT_T, Dim, Idx>;

using BufAccFH2D = alpaka::Buf<Acc, FH2D, Dim, Idx>;
using BufAccFHs = alpaka::Buf<Acc, FHs, Dim, Idx>;
using BufHostFH2D = alpaka::Buf<Host, FH2D, Dim, Idx>;
using BufHostFHs = alpaka::Buf<Host, FHs, Dim, Idx>;

#endif
