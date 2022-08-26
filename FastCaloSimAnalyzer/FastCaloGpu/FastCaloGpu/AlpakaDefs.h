/**
 **/

#ifndef ALPAKADEFS_H
#define ALPAKADEFS_H

#include <alpaka/alpaka.hpp>
#include "GeoGpu_structs.h"
#include "GpuGeneral_structs.h"

using Dim = alpaka::DimInt<1>;
using Idx = std::size_t;
using Vec = alpaka::Vec<Dim, Idx>;
using Acc = alpaka::AccGpuCudaRt<Dim, Idx>;
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

using BufAccCaloDDE = alpaka::Buf<Acc, CaloDetDescrElement, Dim, Idx>;
using BufAccSampleIndex = alpaka::Buf<Acc, Rg_Sample_Index, Dim, Idx>;
using BufAccGeoRegion  = alpaka::Buf<Acc, GeoRegion, Dim, Idx>;
using BufAccGeoGpu = alpaka::Buf<Acc, GeoGpu, Dim, Idx>;
using BufAccLongLong = alpaka::Buf<Acc, long long, Dim, Idx>;

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
using CellCtT = alpaka::Buf<Acc, CELL_CT_T, Dim, Idx>;

#endif
