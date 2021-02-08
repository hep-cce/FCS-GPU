#include "Rand4Hits.h"
#include "gpuQ.h"
#include <iostream>
#include <curand.h>

#include "Rand4Hits_cpu.cxx"

#define CURAND_CALL( x )                                                                                               \
  if ( ( x ) != CURAND_STATUS_SUCCESS ) {                                                                              \
    printf( "Error at %s:%d\n", __FILE__, __LINE__ );                                                                  \
    exit( EXIT_FAILURE );                                                                                              \
  }

void Rand4Hits::allocate_simulation( long long /*maxhits*/, unsigned short /*maxbins*/, unsigned short maxhitct,
                                     unsigned long n_cells ) {

  float* Cells_Energy;
  int*   ct;
  gpuQ( cudaMalloc( (void**)&Cells_Energy, n_cells * sizeof( float ) ) );
  m_cells_energy = Cells_Energy;
  Cell_E* cell_e;
  gpuQ( cudaMalloc( (void**)&cell_e, maxhitct * sizeof( Cell_E ) ) );
  m_cell_e   = cell_e;
  m_cell_e_h = (Cell_E*)malloc( maxhitct * sizeof( Cell_E ) );
  gpuQ( cudaMalloc( (void**)&ct, sizeof( int ) ) );
  m_ct = ct;
}

Rand4Hits::~Rand4Hits() {
  gpuQ( cudaFree( m_rand_ptr ) );
  if ( m_useCPU ) {
    destroyCPUGen();
  } else {
    CURAND_CALL( curandDestroyGenerator( *( (curandGenerator_t*)m_gen ) ) );
    delete (curandGenerator_t*)m_gen;
  }
};

void Rand4Hits::rd_regen() {
  if ( m_useCPU ) {
    genCPU( 3 * m_total_a_hits );
    gpuQ( cudaMemcpy( m_rand_ptr, m_rnd_cpu.data(), 3 * m_total_a_hits * sizeof( float ), cudaMemcpyHostToDevice ) );
  } else {
    CURAND_CALL( curandGenerateUniform( *( (curandGenerator_t*)m_gen ), m_rand_ptr, 3 * m_total_a_hits ) );
  }
};

void Rand4Hits::create_gen( unsigned long long seed, size_t num, bool useCPU ) {

  float* f{nullptr};
  gpuQ( cudaMalloc( &f, num * sizeof( float ) ) );

  m_useCPU = useCPU;
  
  if ( m_useCPU ) {
    createCPUGen( seed );
    genCPU( num );
    gpuQ( cudaMemcpy( f, m_rnd_cpu.data(), num * sizeof( float ), cudaMemcpyHostToDevice ) );
  } else {
    curandGenerator_t* gen = new curandGenerator_t;
    CURAND_CALL( curandCreateGenerator( gen, CURAND_RNG_PSEUDO_DEFAULT ) );
    CURAND_CALL( curandSetPseudoRandomGeneratorSeed( *gen, seed ) );
    CURAND_CALL( curandGenerateUniform( *gen, f, num ) );
    m_gen = (void*)gen;
  }

  m_rand_ptr = f;
}
