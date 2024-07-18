/*
  Copyright (C) 2002-2021 CERN for the benefit of the ATLAS collaboration
*/

#include "Rand4Hits.h"
#include "gpuQ.h"
#include <iostream>

#ifndef RNDGEN_CPU
#if defined (HIP_TARGET_NVIDIA)
#include <curand.h>
#else
#include <hiprand.h>
#endif
#endif

#include "Rand4Hits_cpu.cxx"

#if defined (HIP_TARGET_NVIDIA)
  #define CURAND_CALL(x)                                                         \
    if ((x) != CURAND_STATUS_SUCCESS) {                                          \
      printf("Error at %s:%d\n", __FILE__, __LINE__);                            \
      exit(EXIT_FAILURE);                                                        \
    }
#else
  #define CURAND_CALL( x )                                                                                               \
    if ( ( x ) != HIPRAND_STATUS_SUCCESS ) {                                                                              \
      printf( "Error at %s:%d\n", __FILE__, __LINE__ );                                                                  \
      exit( EXIT_FAILURE );                                                                                              \
    }
#endif

#ifndef USE_STDPAR
void Rand4Hits::allocate_simulation( long long /*maxhits*/, unsigned short /*maxbins*/, unsigned short maxhitct,
                                     unsigned long n_cells ) {

  // for args.cells_energy
  CELL_ENE_T* Cells_Energy;
  gpuQ( hipMalloc( (void**)&Cells_Energy, n_cells * sizeof( CELL_ENE_T ) ) );
  m_cells_energy = Cells_Energy;

  // for args.hitcells_E
  Cell_E* cell_e;
  gpuQ( hipMalloc( (void**)&cell_e, maxhitct * sizeof( Cell_E ) ) );
  m_cell_e   = cell_e;
  m_cell_e_h = (Cell_E*)malloc( maxhitct * sizeof( Cell_E ) );

  // for args.hitcells_E_h and args.hitcells_ct
  int*   ct;
  gpuQ( hipMalloc( (void**)&ct, sizeof( int ) ) );
  m_ct = ct;

  printf(" -- R4H ncells: %lu  cells_energy: %p   hitcells_E: %p  hitcells_ct: %p\n",
         n_cells, (void*)m_cells_energy, (void*)m_cell_e, (void*)m_ct);

}
#endif


#ifndef USE_STDPAR
void Rand4Hits::allocateGenMem(size_t num) {
  m_rnd_cpu = new std::vector<float>;
  m_rnd_cpu->resize(num);
  std::cout << "m_rnd_cpu: " << m_rnd_cpu << "  " << m_rnd_cpu->data() << std::endl;
}
#endif


Rand4Hits::~Rand4Hits() {
  
#ifdef USE_STDPAR
  deallocate();
#else
  delete ( m_rnd_cpu );
#endif
  
#ifdef USE_STDPAR
  if (!m_useCPU) {
    gpuQ( hipFree( m_rand_ptr ) );
  }
#else
  gpuQ( hipFree( m_rand_ptr ) );
#endif
  
  if ( m_useCPU ) {
    destroyCPUGen();
  } else {
#ifndef RNDGEN_CPU
#if defined (HIP_TARGET_NVIDIA)
    CURAND_CALL(curandDestroyGenerator(*((curandGenerator_t *)m_gen)));
    delete (curandGenerator_t *)m_gen;
#else	  
    CURAND_CALL( hiprandDestroyGenerator( *( (hiprandGenerator_t*)m_gen ) ) );
    delete (hiprandGenerator_t*)m_gen;
#endif
#endif
  }
};

void Rand4Hits::rd_regen() {
  if ( m_useCPU ) {
    genCPU( 3 * m_total_a_hits );
    #if defined _NVHPC_STDPAR_GPU || !defined USE_STDPAR
    gpuQ( hipMemcpy( m_rand_ptr, m_rnd_cpu->data(), 3 * m_total_a_hits * sizeof( float ), hipMemcpyHostToDevice ) );
    #endif
  } else {
#ifndef RNDGEN_CPU
#if defined (HIP_TARGET_NVIDIA)
    CURAND_CALL(curandGenerateUniform(*((curandGenerator_t *)m_gen), m_rand_ptr,
                                      3 * m_total_a_hits));
#else
    CURAND_CALL( hiprandGenerateUniform( *( (hiprandGenerator_t*)m_gen ), m_rand_ptr, 3 * m_total_a_hits ) );
#endif
#endif
  }
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
    gpuQ( hipMalloc( &f, num * sizeof( float ) ) );
    gpuQ( hipMemcpy( f, m_rnd_cpu->data(), num * sizeof( float ), hipMemcpyHostToDevice ) );
#endif
  } else {
#ifndef RNDGEN_CPU
    gpuQ( hipMalloc( &f, num * sizeof( float ) ) );
#if defined (HIP_TARGET_NVIDIA)
    curandGenerator_t *gen = new curandGenerator_t;
    CURAND_CALL(curandCreateGenerator(gen, CURAND_RNG_PSEUDO_DEFAULT));
    CURAND_CALL(curandSetPseudoRandomGeneratorSeed(*gen, seed));
    CURAND_CALL(curandGenerateUniform(*gen, f, num));
#else
    hiprandGenerator_t* gen = new hiprandGenerator_t;
    CURAND_CALL( hiprandCreateGenerator( gen, HIPRAND_RNG_PSEUDO_DEFAULT ) );
    CURAND_CALL( hiprandSetPseudoRandomGeneratorSeed( *gen, seed ) );
    CURAND_CALL( hiprandGenerateUniform( *gen, f, num ) );
#endif
    m_gen = (void*)gen;
#endif
  }
  
  m_rand_ptr = f;
  
  std::cout << "R4H m_rand_ptr: " << m_rand_ptr << std::endl;

}
