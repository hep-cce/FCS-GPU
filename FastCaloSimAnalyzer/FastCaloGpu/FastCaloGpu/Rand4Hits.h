#ifndef RAND4HITS_H
#define RAND4HITS_H

#include <curand.h>
#include <stdio.h>

#include "GpuGeneral_structs.h"
#include "gpuQ.h"

#define CURAND_CALL( x )                                                                                               \
  if ( ( x ) != CURAND_STATUS_SUCCESS ) {                                                                              \
    printf( "Error at %s:%d\n", __FILE__, __LINE__ );                                                                  \
    exit( EXIT_FAILURE );                                                                                              \
  }

class Rand4Hits {
public:
  Rand4Hits() {
    m_rand_ptr     = 0;
    m_total_a_hits = 0;
  };
  ~Rand4Hits() {
    gpuQ( cudaFree( m_rand_ptr ) );
    CURAND_CALL( curandDestroyGenerator( m_gen ) );
  };
  // float *  HitsRandGen(unsigned int nhits, unsigned long long seed ) ;

  float* rand_ptr( int nhits ) {
    if ( over_alloc( nhits ) ) {
      rd_regen();
      return m_rand_ptr;
    } else {
      float* f_ptr = &( m_rand_ptr[3 * m_current_hits] );
      return f_ptr;
    }
  };
  float*            rand_ptr_base() { return m_rand_ptr; }
  void              set_rand_ptr( float* ptr ) { m_rand_ptr = ptr; };
  void              set_t_a_hits( int nhits ) { m_total_a_hits = nhits; };
  void              set_c_hits( int nhits ) { m_current_hits = nhits; };
  unsigned int      get_c_hits() { return m_current_hits; };
  unsigned int      get_t_a_hits() { return m_total_a_hits; };
  void              set_gen( curandGenerator_t gen ) { m_gen = gen; };
  curandGenerator_t gen() { return m_gen; };

  void allocate_simulation( long long maxhits, unsigned short maxbins, unsigned short maxhitct, unsigned long n_cells );

  float*  get_cells_energy() { return m_cells_energy; };
  Cell_E* get_cell_e() { return m_cell_e; };
  Cell_E* get_cell_e_h() { return m_cell_e_h; };

  int* get_ct() { return m_ct; };

  unsigned long* get_hitcells() { return m_hitcells; };
  int*           get_hitcells_ct() { return m_hitcells_ct; };

  void rd_regen() { CURAND_CALL( curandGenerateUniform( m_gen, m_rand_ptr, 3 * m_total_a_hits ) ); };
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
  float*            m_rand_ptr;
  unsigned int      m_total_a_hits;
  unsigned int      m_current_hits;
  curandGenerator_t m_gen;

  // patch in some GPU pointers for cudaMalloc
  float*  m_cells_energy;
  Cell_E* m_cell_e;
  int*    m_ct;

  // host side ;
  unsigned long* m_hitcells;
  int*           m_hitcells_ct;
  Cell_E*        m_cell_e_h;
};

#endif
