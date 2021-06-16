#include "Rand4Hits.h"

void Rand4Hits::allocate_simulation( long long maxhits, unsigned short maxbins, unsigned short maxhitct,
                                     unsigned long n_cells ) {

  float* Cells_Energy;
  int*   ct;
  gpuQ( hipMalloc( (void**)&Cells_Energy, n_cells * sizeof( float ) ) );
  m_cells_energy = Cells_Energy;
  Cell_E* cell_e;
  gpuQ( hipMalloc( (void**)&cell_e, maxhitct * sizeof( Cell_E ) ) );
  m_cell_e   = cell_e;
  m_cell_e_h = (Cell_E*)malloc( maxhitct * sizeof( Cell_E ) );
  gpuQ( hipMalloc( (void**)&ct, sizeof( int ) ) );
  m_ct = ct;
}
