/*
  Copyright (C) 2002-2021 CERN for the benefit of the ATLAS collaboration
*/

#include "Rand4Hits.h"
#include "gpuQ.h"
#include "DEV_BigMem.h"

#include <iostream>
#include <curand.h>

#include "GpuParams.h"
#include "Rand4Hits_cpu.cxx"

#define CURAND_CALL(x)                                                         \
  if ((x) != CURAND_STATUS_SUCCESS) {                                          \
    printf("Error at %s:%d\n", __FILE__, __LINE__);                            \
    exit(EXIT_FAILURE);                                                        \
  }

#ifndef USE_STDPAR
void Rand4Hits::allocate_simulation(int maxbins, int maxhitct,
                                    unsigned long n_cells) {

  // for args.cells_energy
  CELL_ENE_T *Cells_Energy;
  gpuQ(cudaMalloc((void **)&Cells_Energy,
                  MAX_SIM * n_cells * sizeof(CELL_ENE_T)));
  m_cells_energy = Cells_Energy;

  // for args.hitcells_E
  Cell_E *cell_e;
  gpuQ(cudaMalloc((void **)&cell_e, MAX_SIM * maxhitct * sizeof(Cell_E)));
  m_cell_e = cell_e;
  m_cell_e_h = (Cell_E *)malloc(MAX_SIM * maxhitct * sizeof(Cell_E));

  long *simbins;
  gpuQ(cudaMalloc((void **)&simbins, MAX_SIMBINS * sizeof(long)));
  m_simbins = simbins;

  HitParams *hitparams;
  gpuQ(cudaMalloc((void **)&hitparams, MAX_SIMBINS * sizeof(HitParams)));
  m_hitparams = hitparams;

  int *ct_ptr;
  gpuQ(cudaMalloc((void **)&ct_ptr, MAX_SIM * sizeof(int)));
  m_ct = ct_ptr;
  m_ct_h = (int *)malloc(MAX_SIM * sizeof(int));

  DEV_BigMem *bm = new DEV_BigMem(M_SEG_SIZE);
  DEV_BigMem::bm_ptr = bm;

  printf(" -- R4H ncells: %lu  cells_energy: %p   hitcells_E: %p  hitcells_ct: "
         "%p\n",
         n_cells, (void *)m_cells_energy, (void *)m_cell_e, (void *)m_ct);
}
#endif

#ifndef USE_STDPAR
void Rand4Hits::allocateGenMem(size_t num) {
  m_rnd_cpu = new std::vector<float>;
  m_rnd_cpu->resize(num);
  std::cout << "m_rnd_cpu: " << m_rnd_cpu << "  " << m_rnd_cpu->data()
            << std::endl;
}
#endif

Rand4Hits::~Rand4Hits() {

#ifdef USE_STDPAR
  deallocate();
#else
  delete (m_rnd_cpu);
  if (DEV_BigMem::bm_ptr) {
    std::cout << "BigMem allocated: " << DEV_BigMem::bm_ptr->size()
              << "  used: " << DEV_BigMem::bm_ptr->used()
              << "  lost: " << DEV_BigMem::bm_ptr->lost() << std::endl;
    delete DEV_BigMem::bm_ptr;
  }
#endif

#ifdef USE_STDPAR
  if (!m_useCPU) {
    gpuQ(cudaFree(m_rand_ptr));
  }
#else
  gpuQ(cudaFree(m_rand_ptr));
#endif

  if (m_useCPU) {
    destroyCPUGen();
  } else {
    CURAND_CALL(curandDestroyGenerator(*((curandGenerator_t *)m_gen)));
    delete (curandGenerator_t *)m_gen;
  }
};

void Rand4Hits::rd_regen() {
  if (m_useCPU) {
    genCPU(3 * m_total_a_hits);
#if defined _NVHPC_STDPAR_GPU || !defined USE_STDPAR
    gpuQ(cudaMemcpy(m_rand_ptr, m_rnd_cpu->data(),
                    3 * m_total_a_hits * sizeof(float),
                    cudaMemcpyHostToDevice));
#endif
  } else {
    CURAND_CALL(curandGenerateUniform(*((curandGenerator_t *)m_gen), m_rand_ptr,
                                      3 * m_total_a_hits));
  }
};

void Rand4Hits::create_gen(unsigned long long seed, size_t num, bool useCPU) {

  float *f{ nullptr };

  m_useCPU = useCPU;

  if (m_useCPU) {
    allocateGenMem(num);
    createCPUGen(seed);
    genCPU(num);
#ifdef USE_STDPAR
    f = m_rnd_cpu->data();
#else
    gpuQ(cudaMalloc(&f, num * sizeof(float)));
    gpuQ(cudaMemcpy(f, m_rnd_cpu->data(), num * sizeof(float),
                    cudaMemcpyHostToDevice));
#endif
  } else {
    gpuQ(cudaMalloc(&f, num * sizeof(float)));
    curandGenerator_t *gen = new curandGenerator_t;
    CURAND_CALL(curandCreateGenerator(gen, CURAND_RNG_PSEUDO_DEFAULT));
    CURAND_CALL(curandSetPseudoRandomGeneratorSeed(*gen, seed));
    CURAND_CALL(curandGenerateUniform(*gen, f, num));
    m_gen = (void *)gen;
  }

  m_rand_ptr = f;

  std::cout << "R4H m_rand_ptr: " << m_rand_ptr << std::endl;
}
