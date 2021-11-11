/*
  Copyright (C) 2002-2021 CERN for the benefit of the ATLAS collaboration
*/

#include "GeoLoadGpu.h"
#include "gpuQ.h"
#include "cuda_runtime.h"

bool GeoLoadGpu::LoadGpu_sp() {

  if ( !m_cells || m_ncells == 0 ) {
    std::cout << "Geometry is empty " << std::endl;
    return false;
  }

  if (m_geo_d == 0) {
    m_geo_d = new GeoGpu;
  }
  

  cudaDeviceProp prop;
  gpuQ( cudaGetDeviceProperties( &prop, 0 ) );
  std::cout << "Executing on GPU: " << prop.name << std::endl;
  
  // Allocate Device memory for cells and copy cells as array
  // move cells on host to a array first
  
  // CaloDetDescrElement* cells_Host = (CaloDetDescrElement*)malloc( m_ncells * sizeof( CaloDetDescrElement ) );
  // m_cellid_array                  = (Identifier*)malloc( m_ncells * sizeof( Identifier ) );

  CaloDetDescrElement* cells_Host = new CaloDetDescrElement[m_ncells];
  m_cellid_array = new Identifier[m_ncells];

  
  // create an array of cell identities, they are in order of hashids.
  int ii = 0;
  for ( t_cellmap::iterator ic = m_cells->begin(); ic != m_cells->end(); ++ic ) {
    cells_Host[ii]     = *( *ic ).second;
    Identifier id      = ( ( *ic ).second )->identify();
    m_cellid_array[ii] = id;
    ii++;
  }

  m_cells_d = cells_Host;
  
  //  free( cells_Host ); FIXME!

  if ( 0 ) {
    if ( !SanityCheck() ) { return false; }
  }
  
  // each Region allocate a grid (long Long) gpu array
  //  copy array to GPU
  //  save to regions m_cell_g ;
  for ( unsigned int ir = 0; ir < m_nregions; ++ir ) {
    //	std::cout << "debug m_regions_d[ir].cell_grid()[0] " << m_regions[ir].cell_grid()[0] <<std::endl;
    long long* ptr_g;

    ptr_g = m_regions[ir].cell_grid();
    m_regions[ir].set_cell_grid_g( ptr_g );
    m_regions[ir].set_all_cells( m_cells_d ); // set this so all region instance know where the GPU cells are, before
  }

  // GPU allocate Regions data  and load them to GPU as array of regions


  m_regions_d = m_regions;

  m_geo_d->cells        = m_cells_d;
  m_geo_d->ncells       = m_ncells;
  m_geo_d->nregions     = m_nregions;
  m_geo_d->regions      = m_regions_d;
  m_geo_d->max_sample   = m_max_sample;
  m_geo_d->sample_index = m_sample_index_h;

  std::cout << "STDPAR GEO\n";
  std::cout << "ncells: " << m_geo_d->ncells << "\n";
  // for (int i=0; i<m_geo_d->ncells; ++i) {
  //   CaloDetDescrElement& d= m_geo_d->cells[i];
  //   std::cout << "  " << i << " " << d.m_identify << "  " << d.m_r << "  " << d.m_eta << "\n";
  // }
  std::cout << "regions: " << m_nregions << "\n";
  // for (int i=0; i<m_nregions; ++i) {
  //   GeoRegion& g=m_geo_d->regions[i];
  //   std::cout << "  " << i << " " << g.index() << "  " << g.cell_grid_eta()
  //             << "  " << g.maxphi() << "\n";
  // }

  std::cout << "=================================\n";
  
  return true;
}

