/*
  Copyright (C) 2002-2021 CERN for the benefit of the ATLAS collaboration
*/
#include "Rand4Hits.h"
#include "DEV_BigMem.h"

#include <omp.h>
#ifdef OMP_OFFLOAD_TARGET_NVIDIA
#include "gpuQ.h"
#include <cuda_runtime_api.h>
#include <curand.h>
#elif defined OMP_OFFLOAD_TARGET_AMD
#include "hip/hip_runtime.h"
#include <rocrand.h>
#endif

#include "GpuParams.h"
#include "Rand4Hits_cpu.cxx"

#ifdef OMP_OFFLOAD_TARGET_NVIDIA
#define CURAND_CALL( x )                                                                                               \
  if ( ( x ) != CURAND_STATUS_SUCCESS ) {                                                                              \
    printf( "Error at %s:%d\n", __FILE__, __LINE__ );                                                                  \
    exit( EXIT_FAILURE );                                                                                              \
  }
#elif defined OMP_OFFLOAD_TARGET_AMD
#define ROCRAND_CALL( x )                                                         \
  if ((x) != ROCRAND_STATUS_SUCCESS) {                                          \
    printf("Error at %s:%d\n", __FILE__, __LINE__);                            \
    exit(EXIT_FAILURE);                                                        \
  }
#endif

void Rand4Hits::allocate_simulation( int maxbins, int maxhitct, unsigned long n_cells ) {

  int m_default_device = omp_get_default_device();
  int m_initial_device = omp_get_initial_device();

  CELL_ENE_T* Cells_Energy;
  Cells_Energy   = (float*)omp_target_alloc( MAX_SIM * n_cells * sizeof( CELL_ENE_T ), m_default_device );
  m_cells_energy = Cells_Energy;

  Cell_E* cell_e;
  cell_e     = (Cell_E*)omp_target_alloc( MAX_SIM * maxhitct * sizeof( Cell_E ), m_default_device );
  m_cell_e   = cell_e;
  m_cell_e_h = (Cell_E*)malloc( MAX_SIM * maxhitct * sizeof( Cell_E ) );

  long* simbins;
  simbins   = (long*)omp_target_alloc( MAX_SIMBINS * sizeof( long ), m_default_device );
  m_simbins = simbins;

  HitParams* hitparams;
  hitparams   = (HitParams*)omp_target_alloc( MAX_SIMBINS * sizeof( HitParams ), m_default_device );
  m_hitparams = hitparams;

  int* ct_ptr;
  ct_ptr = (int*)omp_target_alloc( MAX_SIM * sizeof( int ), m_default_device );
  m_ct   = ct_ptr;
  m_ct_h = (int*)malloc( MAX_SIM * sizeof( int ) );

  DEV_BigMem* bm     = new DEV_BigMem( M_SEG_SIZE );
  DEV_BigMem::bm_ptr = bm;

  printf( " -- R4H ncells: %lu  cells_energy: %p   hitcells_E: %p  hitcells_ct: "
          "%p\n",
          n_cells, (void*)m_cells_energy, (void*)m_cell_e, (void*)m_ct );
}

void Rand4Hits::allocateGenMem( size_t num ) {
  m_rnd_cpu = new std::vector<float>;
  m_rnd_cpu->resize( num );
  std::cout << "m_rnd_cpu: " << m_rnd_cpu << "  " << m_rnd_cpu->data() << std::endl;
}

Rand4Hits::~Rand4Hits() {

  delete ( m_rnd_cpu );
  if ( DEV_BigMem::bm_ptr ) {
    std::cout << "BigMem allocated: " << DEV_BigMem::bm_ptr->size() << "  used: " << DEV_BigMem::bm_ptr->used()
              << "  lost: " << DEV_BigMem::bm_ptr->lost() << std::endl;
    delete DEV_BigMem::bm_ptr;
  }
  omp_target_free( m_rand_ptr, m_select_device );
  if ( m_useCPU ) {
    destroyCPUGen();
  } else {
#ifndef RNDGEN_CPU
#ifdef OMP_OFFLOAD_TARGET_NVIDIA
    CURAND_CALL( curandDestroyGenerator( *( (curandGenerator_t*)m_gen ) ) );
    delete (curandGenerator_t*)m_gen;
#elif defined OMP_OFFLOAD_TARGET_AMD
    ROCRAND_CALL(rocrand_destroy_generator( *( (rocrand_generator*)m_gen)));
    delete (rocrand_generator *)m_gen;
#endif
#endif
  }
};

void Rand4Hits::rd_regen() {
  if ( m_useCPU ) {
    genCPU( 3 * m_total_a_hits );
    if ( omp_target_memcpy( m_rand_ptr, m_rnd_cpu->data(), 3 * m_total_a_hits * sizeof( float ), m_offset, m_offset,
                            m_select_device, m_initial_device ) ) {
      std::cout << "ERROR: copy random numbers from cpu to gpu " << std::endl;
    }
  } else {
#ifndef RNDGEN_CPU
#ifdef OMP_OFFLOAD_TARGET_NVIDIA
    CURAND_CALL( curandGenerateUniform( *( (curandGenerator_t*)m_gen ), m_rand_ptr, 3 * m_total_a_hits ) );
#elif defined OMP_OFFLOAD_TARGET_AMD
    ROCRAND_CALL(rocrand_generate_uniform( *( (rocrand_generator*)m_gen), m_rand_ptr, 3 * m_total_a_hits));
#endif
#endif
  }
};

void Rand4Hits::create_gen( unsigned long long seed, size_t num, bool useCPU ) {

  float* f{ nullptr };

  m_useCPU = useCPU;

  if ( m_useCPU ) {
    allocateGenMem( num );
    createCPUGen( seed );
    genCPU( num );
    f = (float*)omp_target_alloc( num * sizeof( float ), m_select_device );
    if ( f == NULL ) { std::cout << " ERROR: No space left on device." << std::endl; }
    if ( omp_target_memcpy( f, m_rnd_cpu->data(), num * sizeof( float ), m_offset, m_offset, m_select_device,
                            m_initial_device ) ) {
      std::cout << "ERROR: copy random numbers from cpu to gpu " << std::endl;
    }
  } else {
#ifndef RNDGEN_CPU
#ifdef OMP_OFFLOAD_TARGET_NVIDIA
    gpuQ( cudaMalloc( &f, num * sizeof( float ) ) );
    curandGenerator_t* gen = new curandGenerator_t;
    CURAND_CALL( curandCreateGenerator( gen, CURAND_RNG_PSEUDO_DEFAULT ) );
    CURAND_CALL( curandSetPseudoRandomGeneratorSeed( *gen, seed ) );
    CURAND_CALL( curandGenerateUniform( *gen, f, num ) );
    m_gen = (void*)gen;
#elif defined OMP_OFFLOAD_TARGET_AMD
    hipMalloc(&f, num * sizeof(float));
    rocrand_generator* gen = new rocrand_generator;
    ROCRAND_CALL(rocrand_create_generator(gen, ROCRAND_RNG_PSEUDO_DEFAULT));
    ROCRAND_CALL(rocrand_set_seed(*gen, seed));
    ROCRAND_CALL(rocrand_generate_uniform(*gen, f, num));
    m_gen = (void*)gen;
#endif
#endif
  }

  m_rand_ptr = f;

  std::cout << "R4H m_rand_ptr: " << m_rand_ptr << std::endl;
}
