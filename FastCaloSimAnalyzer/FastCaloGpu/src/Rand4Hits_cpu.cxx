/*
  Copyright (C) 2002-2021 CERN for the benefit of the ATLAS collaboration
*/

#include "Rand4Hits.h"

#include <random>
#include <vector>
#include <algorithm>
#include <iostream>

#define cpu_randgen_t std::mt19937

void Rand4Hits::createCPUGen( unsigned long long seed ) {
  cpu_randgen_t* eng = new cpu_randgen_t( seed );
  m_gen              = (void*)eng;
}

void Rand4Hits::destroyCPUGen() {
  if ( m_gen ) { delete (cpu_randgen_t*)m_gen; }
  //  if ( m_rnd_cpu ) { delete (m_rnd_cpu); }
}

float* Rand4Hits::genCPU( size_t num ) {
  
  m_rnd_cpu->resize( num );

  cpu_randgen_t* eng = (cpu_randgen_t*)( m_gen );

  auto RNG = [eng]( float low, float high ) {
    auto randomFunc = [distribution_  = std::uniform_real_distribution<float>( low, high ),
                       random_engine_ = *eng]() mutable { return distribution_( random_engine_ ); };
    return randomFunc;
  };

  std::generate_n( m_rnd_cpu->begin(), num, RNG( 0.f, 1.f ) );

  return m_rnd_cpu->data();
}

Rand4Hits::~Rand4Hits() {
#ifdef USE_STDPAR
  deallocate();
#else
  delete ( m_rnd_cpu );
#endif

  destroyCPUGen();
}

void Rand4Hits::rd_regen() {
    genCPU( 3 * m_total_a_hits );
};

void Rand4Hits::create_gen( unsigned long long seed, size_t num, bool useCPU ) {

  float* f{nullptr};

  m_useCPU = useCPU;
  
  if ( m_useCPU ) {
    allocateGenMem( num );
    createCPUGen( seed );
    genCPU( num );
#ifdef USE_STDPAR
    f = m_rnd_cpu->data();
#else
    gpuQ( cudaMalloc( &f, num * sizeof( float ) ) );
    gpuQ( cudaMemcpy( f, m_rnd_cpu->data(), num * sizeof( float ), cudaMemcpyHostToDevice ) );
#endif
  } else {
    std::cout << "ERROR: should only be using CPU for Random Number Generator\n";
    exit(1);
  }
  
  m_rand_ptr = f;
  
  std::cout << "R4H m_rand_ptr: " << m_rand_ptr << std::endl;

}
