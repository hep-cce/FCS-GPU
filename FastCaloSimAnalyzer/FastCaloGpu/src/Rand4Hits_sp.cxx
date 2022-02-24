#include "Rand4Hits.h"
#include "GpuParams.h"
#include <iostream>
#include <cstring>

#include <execution>
#include <algorithm>
#include "CountingIterator.h"


void Rand4Hits::allocate_simulation( int maxbins, int maxhitct, unsigned long n_cells ) {

  std::cout << "R4H::allocate_simulation\n";
  float *f = new float[100];
  std::for_each_n(std::execution::par_unseq, counting_iterator(0), 100,
                  [=](unsigned int tid) {
                    f[tid] = tid;
                  }
                  );
  std::cout << "f[33] = " << f[33] << std::endl;
  


  
  // for args.cells_energy
  //  m_cells_energy = (CELL_ENE_T*)malloc( MAX_SIM * n_cells * sizeof(CELL_ENE_T) );
  m_cells_energy = new CELL_ENE_T[MAX_SIM * n_cells];


  std::atomic<unsigned long>* cene = new std::atomic<unsigned long>[MAX_SIM * n_cells];
  std::cout << "clearing ene " << MAX_SIM*n_cells << " at " << (void*) cene << std::endl;
  std::for_each_n(std::execution::par_unseq, counting_iterator(0), MAX_SIM*n_cells,
                  [=](unsigned int tid) {
                    printf(" %u %lu\n",tid, (unsigned int)cene[tid]);
                    //                    cene[tid].store(0.0);
                  }
                  );     

  

  // for args.hitcells_E
  m_cell_e_h = (Cell_E*)malloc( MAX_SIM * maxhitct * sizeof( Cell_E ) );

  // for args.hitcells_E_h and args.hitcells_ct
  m_cell_e = m_cell_e_h;

  m_simbins = (long*)std::malloc( MAX_SIMBINS * sizeof( long ));

  m_hitparams = (HitParams*)std::malloc( MAX_SIMBINS * sizeof( HitParams ) );

  m_ct = new std::atomic<int>[MAX_SIM];
  m_ct_h = new int[MAX_SIM];

  printf(" R4H_sp ncells: %lu  cells_energy: %p  cells_size: %lu hitcells_E: %p  hitcells_ct: %p\n",
         n_cells, (void*)m_cells_energy, MAX_SIM*n_cells, (void*)m_cell_e, (void*)m_ct);
  
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

