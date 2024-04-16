#include "Rand4Hits.h"
#include <iostream>
#include <cstring>

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

