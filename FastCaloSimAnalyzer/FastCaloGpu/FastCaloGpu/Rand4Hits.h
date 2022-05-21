/*
  Copyright (C) 2002-2021 CERN for the benefit of the ATLAS collaboration
*/

#ifndef RAND4HITS_H
#define RAND4HITS_H

#include <vector>
#include <random>

#ifdef USE_KOKKOS
#  include <Kokkos_Core.hpp>
#  include <Kokkos_Random.hpp>
#endif

#ifdef USE_STDPAR
#   include <atomic>
#endif

#ifdef USE_ALPAKA

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
// Alpaka examples
template<typename TAcc>
using RandomEngine = alpaka::rand::Philox4x32x10<TAcc>;

using BufHostEngine = alpaka::Buf<Host, RandomEngine<Acc>, Dim, Idx>;
using BufAccEngine = alpaka::Buf<Acc, RandomEngine<Acc>, Dim, Idx>;

// The choice of NUM_STATES value is totally random
unsigned constexpr NUM_STATES = 1000;
#endif

#include "GpuGeneral_structs.h"

class Rand4Hits {
public:  

#ifdef USE_ALPAKA
  Rand4Hits()
    : m_queue(alpaka::getDevByIdx<Acc>(Idx{0}))
    , m_bufAcc(alpaka::allocBuf<float, Idx>(alpaka::getDevByIdx<Acc>(0u), Vec{Idx(1u)}))
    , m_bufAccEngine(alpaka::allocBuf<RandomEngine<Acc>, Idx>(alpaka::getDevByIdx<Acc>(0u), Vec{Idx(NUM_STATES)}))
    {}
#else
  Rand4Hits() = default;
#endif
  ~Rand4Hits();

  float* rand_ptr( int nhits ) {
    if ( over_alloc( nhits ) ) {
      rd_regen();
      return m_rand_ptr;
    } else {
      float* f_ptr = &( m_rand_ptr[3 * m_current_hits] );
      return f_ptr;
    }
  };
  float*       rand_ptr_base() { return m_rand_ptr; }
  void         set_rand_ptr( float* ptr ) { m_rand_ptr = ptr; };
  void         set_t_a_hits( int nhits ) { m_total_a_hits = nhits; };
  void         set_c_hits( int nhits ) { m_current_hits = nhits; };
  unsigned int get_c_hits() { return m_current_hits; };
  unsigned int get_t_a_hits() { return m_total_a_hits; };

  void create_gen( unsigned long long seed, size_t numhits, bool useCPU = false );

  void allocate_simulation( long long maxhits, unsigned short maxbins, unsigned short maxhitct, unsigned long n_cells );

#ifdef USE_STDPAR
  void deallocate();
#endif
  
  CELL_ENE_T*  get_cells_energy() { return m_cells_energy; };
  Cell_E* get_cell_e()            { return m_cell_e; };
  Cell_E* get_cell_e_h()          { return m_cell_e_h; };

  CELL_CT_T* get_ct() { return m_ct; };

  unsigned long* get_hitcells()    { return m_hitcells; };
  int*           get_hitcells_ct() { return m_hitcells_ct; };

  void rd_regen();

  void add_a_hits( int nhits ) {
    if ( over_alloc( nhits ) )
      m_current_hits = nhits;
    else
      m_current_hits += nhits;
  };
  bool over_alloc( int nhits ) {
    return m_current_hits + nhits > m_total_a_hits;
  }; // return true if hits over spill, need regenerat rand..

private:
  float* genCPU( size_t num );
  void   createCPUGen( unsigned long long seed );
  void   allocateGenMem( size_t num );
  void   destroyCPUGen();

  float*       m_rand_ptr{nullptr};
  unsigned int m_total_a_hits{0};
  unsigned int m_current_hits;
  void*        m_gen{nullptr};
  bool         m_useCPU{false};

  // patch in some GPU pointers for cudaMalloc
  CELL_ENE_T*  m_cells_energy {nullptr};
  Cell_E*      m_cell_e {nullptr};
  CELL_CT_T*   m_ct {nullptr};

  // host side ;
  unsigned long* m_hitcells{nullptr};
  int*           m_hitcells_ct{nullptr};
  Cell_E*        m_cell_e_h{nullptr};

  std::vector<float>* m_rnd_cpu{nullptr};

#ifdef USE_KOKKOS
  Kokkos::View<float*>  m_cells_energy_v;
  Kokkos::View<Cell_E*> m_cell_e_v;
  Kokkos::View<int>     m_ct_v;
  Kokkos::View<float*>  m_rand_ptr_v;
#endif

#ifdef USE_ALPAKA
  BufAcc m_bufAcc;
  BufAccEngine m_bufAccEngine;
  QueueAcc m_queue;
#endif
};

#endif
