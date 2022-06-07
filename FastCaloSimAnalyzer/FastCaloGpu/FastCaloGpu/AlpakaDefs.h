/**
 **/

#ifndef ALPAKADEFS_H
#define ALPAKADEFS_H

#include <alpaka/alpaka.hpp>

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

// This engine was chosen only because it is used by one of
// the Alpaka examples
template<typename TAcc>
using RandomEngine = alpaka::rand::Philox4x32x10<TAcc>;

using BufHostEngine = alpaka::Buf<Host, RandomEngine<Acc>, Dim, Idx>;
using BufAccEngine = alpaka::Buf<Acc, RandomEngine<Acc>, Dim, Idx>;

// The choice of NUM_STATES value is totally random
unsigned constexpr NUM_STATES = 1000;

#endif
