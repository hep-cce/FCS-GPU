/*
  Copyright (C) 2002-2021 CERN for the benefit of the ATLAS collaboration
*/

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
#ifdef USE_STDPAR
  Cells_Energy = (float*)malloc(n_cells*sizeof(float));
#else
  gpuQ( cudaMalloc( (void**)&Cells_Energy, n_cells * sizeof( float ) ) );
#endif
  m_cells_energy = Cells_Energy;
  
#ifdef USE_STDPAR
#else
  Cell_E* cell_e;
  gpuQ( cudaMalloc( (void**)&cell_e, maxhitct * sizeof( Cell_E ) ) );
  m_cell_e   = cell_e;
#endif
  m_cell_e_h = (Cell_E*)malloc( maxhitct * sizeof( Cell_E ) );
  
#ifdef USE_STDPAR
  m_cell_e = m_cell_e_h;
  m_ct = new std::atomic<int>{0};
#else   
  int*   ct;
  gpuQ( cudaMalloc( (void**)&ct, sizeof( int ) ) );
  m_ct = ct;
#endif
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


  // float *fh = new float[100];
  // cudaMemcpy( fh, f, 10*sizeof(float), cudaMemcpyDeviceToHost );
  // std::cout << "rndptr: " << m_rand_ptr << std::endl;
  // for (int i=0; i<10; ++i) {
  //   //    std::cout << "r4h: " << m_rnd_cpu[i] << std::endl;
  //   std::cout << "r4h: " << fh[i] << std::endl;
  // }
  //  m_rand_ptr = m_rnd_cpu.data();
}
