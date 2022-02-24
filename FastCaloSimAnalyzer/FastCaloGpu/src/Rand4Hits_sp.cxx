#include "Rand4Hits.h"
#include "GpuParams.h"
#include <iostream>
#include <cstring>

void Rand4Hits::allocate_simulation( int maxbins, int maxhitct, unsigned long n_cells ) {


  // for args.cells_energy
  m_cells_energy = (CELL_ENE_T*)malloc( MAX_SIM * n_cells * sizeof(CELL_ENE_T) );

  // for args.hitcells_E
  m_cell_e_h = (Cell_E*)malloc( MAX_SIM * maxhitct * sizeof( Cell_E ) );

  // for args.hitcells_E_h and args.hitcells_ct
  m_cell_e = m_cell_e_h;

  m_simbins = (long*)std::malloc( MAX_SIMBINS * sizeof( long ));

  m_hitparams = (HitParams*)std::malloc( MAX_SIMBINS * sizeof( HitParams ) );

  m_ct = new std::atomic<int>[MAX_SIM];
  m_ct_h = new int[MAX_SIM];

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

