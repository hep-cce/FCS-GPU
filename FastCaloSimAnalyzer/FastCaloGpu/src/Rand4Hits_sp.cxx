#include "Rand4Hits.h"
#include <iostream>
#include <cstring>

#ifndef RNDGEN_CPU
#include "gpuQ.h"
#endif

void Rand4Hits::allocate_simulation( long long /*maxhits*/, unsigned short /*maxbins*/, unsigned short maxhitct,
                                     unsigned long n_cells ) {

  // for args.cells_energy
  m_cells_energy = (CELL_ENE_T*)malloc( n_cells * sizeof(CELL_ENE_T) );

  // for args.hitcells_E
  m_cell_e_h = (Cell_E*)malloc( maxhitct * sizeof( Cell_E ) );

  // for args.hitcells_E_h and args.hitcells_ct
  m_cell_e = m_cell_e_h;
  m_ct     = new CELL_CT_T{0};

  printf(" R4H ncells: %lu  cells_energy: %p   hitcells_E: %p  hitcells_ct: %p\n",
         n_cells, (void*)m_cells_energy, (void*)m_cell_e, (void*)m_ct);
  
}


void Rand4Hits::allocateGenMem(size_t num) {
  m_rnd_cpu = new std::vector<float>;
  m_rnd_cpu->resize(num);
  std::cout << "m_rnd_cpu: " << m_rnd_cpu << "  " << m_rnd_cpu->data() << std::endl;
}


void Rand4Hits::deallocate() {
  free ( m_cells_energy );
  free ( m_cell_e_h );
  free ( m_ct );
  delete ( m_rnd_cpu );
}

/*
**** these are also defined in Rand4Hits.cu
*/
//#ifndef _NVHPC_STDPAR_GPU
#ifdef RNDGEN_CPU

Rand4Hits::~Rand4Hits() {
  deallocate();

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
    throw std::runtime_error( "Rand4Hits::create_gen CPU ERROR: should only be using CPU for Random Number Generator\n" );
  }

  m_rand_ptr = f;

  std::cout << "R4H m_rand_ptr: " << m_rand_ptr << std::endl;

}

#endif
