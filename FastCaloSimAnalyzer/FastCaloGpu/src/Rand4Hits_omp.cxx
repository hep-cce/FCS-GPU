/*
  Copyright (C) 2002-2021 CERN for the benefit of the ATLAS collaboration
*/

#include "Rand4Hits.h"
#include <iostream>
#include <curand.h>

#include "Rand4Hits_cpu.cxx"

void Rand4Hits::allocate_simulation( long long /*maxhits*/, unsigned short /*maxbins*/, unsigned short maxhitct,
                                     unsigned long n_cells ) {

  float* Cells_Energy;
  int*   ct;
//  gpuQ( cudaMalloc( (void**)&Cells_Energy, n_cells * sizeof( float ) ) );
  Cells_Energy = (float *) omp_target_alloc( n_cells * sizeof( float ), m_default_device);
  if ( Cells_Energy == NULL ) {
    std::cout << " ERROR: No space left on device for Cells_Energy." << std::endl;
  }
  m_cells_energy = Cells_Energy;

  Cell_E* cell_e;
//  gpuQ( cudaMalloc( (void**)&cell_e, maxhitct * sizeof( Cell_E ) ) );
  cell_e = (Cell_E *) omp_target_alloc( maxhitct * sizeof( Cell_E ), m_default_device);
  if ( cell_e == NULL ) {
    std::cout << " ERROR: No space left on device for cell_e." << std::endl;
  }
  m_cell_e   = cell_e;

  m_cell_e_h = (Cell_E*)malloc( maxhitct * sizeof( Cell_E ) );
//  gpuQ( cudaMalloc( (void**)&ct, sizeof( int ) ) );
  ct = (int *) omp_target_alloc( sizeof( int ), m_default_device);
  if ( ct == NULL ) {
    std::cout << " ERROR: No space left on device for ct." << std::endl;
  }
  m_ct = ct;
}

Rand4Hits::~Rand4Hits() {
    omp_target_free ( m_rand_ptr, m_default_device );
//  gpuQ( cudaFree( m_rand_ptr ) );
    destroyCPUGen();
//  } else {
//    CURAND_CALL( curandDestroyGenerator( *( (curandGenerator_t*)m_gen ) ) );
//    delete (curandGenerator_t*)m_gen;
//  }
};

void Rand4Hits::rd_regen() {
//  if ( m_useCPU ) {
    genCPU( 3 * m_total_a_hits );
    if ( omp_target_memcpy( m_rand_ptr, m_rnd_cpu.data(), 3 * m_total_a_hits * sizeof( float ), m_offset, m_offset, m_default_device, m_initial_device ) ) {
       std::cout << "ERROR: copy random numbers from cpu to gpu " << std::endl;
    }
//  } else {
//    CURAND_CALL( curandGenerateUniform( *( (curandGenerator_t*)m_gen ), m_rand_ptr, 3 * m_total_a_hits ) );
//  }
};

void Rand4Hits::create_gen( unsigned long long seed, size_t num, bool useCPU ) {

  float* f{nullptr};
  f = (float *) omp_target_alloc( num * sizeof( float ), m_default_device);
  if ( f == NULL ) {
    std::cout << " ERROR: No space left on device." << std::endl;
  }

  m_useCPU = useCPU;
  //always generating and copying from cpu, change later
  //if ( m_useCPU ) {
    createCPUGen( seed );
    genCPU( num );
    if ( omp_target_memcpy( f, m_rnd_cpu.data(), num * sizeof( float ), m_offset, m_offset, m_default_device, m_initial_device ) ) {
       std::cout << "ERROR: copy random numbers from cpu to gpu " << std::endl; 
    }
  //} else {
    curandGenerator_t* gen = new curandGenerator_t;
//    CURAND_CALL( curandCreateGenerator( gen, CURAND_RNG_PSEUDO_DEFAULT ) );
//    CURAND_CALL( curandSetPseudoRandomGeneratorSeed( *gen, seed ) );
//    CURAND_CALL( curandGenerateUniform( *gen, f, num ) );
//    m_gen = (void*)gen;
//  }

  m_rand_ptr = f;
}
