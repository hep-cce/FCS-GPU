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

//#ifdef USE_OMPGPU
#  include <omp.h>
//#endif

#include "GpuGeneral_structs.h"

class Rand4Hits {
public:
  Rand4Hits() {
    m_rand_ptr     = 0;
    m_total_a_hits = 0;
#ifdef USE_OMPGPU
    select_omp_device();
#endif
  };
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
  float        get_cpu_rand_at( long long i) { return m_rnd_cpu.at( i ); }
  float*       rnd_ptr_cpu() { return m_rnd_cpu.data(); }

  void create_gen( unsigned long long seed, size_t numhits, bool useCPU = false );

  void allocate_simulation( long long maxhits, unsigned short maxbins, unsigned short maxhitct, unsigned long n_cells );

  float*  get_cells_energy() { return m_cells_energy; };
  Cell_E* get_cell_e() { return m_cell_e; };
  Cell_E* get_cell_e_h() { return m_cell_e_h; };

  int* get_ct() { return m_ct; };

  unsigned long* get_hitcells() { return m_hitcells; };
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

  void select_omp_device ( ) {
  if ( offload_var == "mandatory" ) 
	m_select_device = m_default_device;
  else if ( offload_var == "disabled" )
      m_select_device = m_initial_device;
  };

private:
  float* genCPU( size_t num );
  void   createCPUGen( unsigned long long seed );
  void   destroyCPUGen();

  float*       m_rand_ptr{nullptr};
  unsigned int m_total_a_hits;
  unsigned int m_current_hits;
  void*        m_gen{nullptr};
  bool         m_useCPU{false};

  // patch in some GPU pointers for cudaMalloc
  float*  m_cells_energy {0};
  Cell_E* m_cell_e {0};
  int*    m_ct {0};

  // host side ;
  unsigned long* m_hitcells{nullptr};
  int*           m_hitcells_ct{nullptr};
  Cell_E*        m_cell_e_h{nullptr};

  std::vector<float> m_rnd_cpu;

#ifdef USE_KOKKOS
  Kokkos::View<float*>  m_cells_energy_v;
  Kokkos::View<Cell_E*> m_cell_e_v;
  Kokkos::View<int>     m_ct_v;
  Kokkos::View<float*>  m_rand_ptr_v;
#endif

#ifdef USE_OMPGPU
  int m_default_device = omp_get_default_device();
  int m_initial_device = omp_get_initial_device();
  std::size_t m_offset = 0;
  const char *env_var = "OMP_TARGET_OFFLOAD";
  std::string offload_var = std::getenv (env_var);
  int m_select_device = m_default_device;
#endif



};

#endif
