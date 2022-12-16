/*
  Copyright (C) 2002-2021 CERN for the benefit of the ATLAS collaboration
*/

#include "GeoLoadGpu.h"

#include <omp.h>
#include <iostream>

bool GeoLoadGpu::LoadGpu_omp() {

  #pragma omp declare mapper(Chain0_Args args) map(to : args.extrapol_eta_ent, \
		  args.extrapol_phi_ent, args.extrapol_r_ent, args.extrapol_z_ent, args.extrapol_eta_ext, \
		  args.extrapol_phi_ext, args.extrapol_r_ext, args.extrapol_z_ext, args.extrapWeight, \
		  args.charge, args.is_phi_symmetric, args.fh2d, args.fhs, args.cs, args.nhits, \
		  args.ncells ) use_by_default

  m_num_devices    = omp_get_num_devices();
  m_initial_device = omp_get_initial_device();
  m_default_device = omp_get_default_device();
  m_offset = 0;
     
  /**
  * Offloading the geometry on the default device
  * using omp_target_alloc and omp_target_memcpy 
  */
  if ( !m_cells || m_ncells == 0 ) {
    std::cout << "Geometry is empty " << std::endl;
    return false;
  }

  // Allocate Device memory for cells and copy cells as array
  // move cells on host to a array first
  m_cells_d = (CaloDetDescrElement *) omp_target_alloc( sizeof( CaloDetDescrElement ) * m_ncells, m_default_device); 
  if ( m_cells_d == NULL ) {
    std::cout << " ERROR: No space left on device." << std::endl;
    return false;
  }

  std::cout << "omp_target_alloc " << m_ncells << " cells" << std::endl;

  CaloDetDescrElement* cells_Host = (CaloDetDescrElement*)malloc( m_ncells * sizeof( CaloDetDescrElement ) );
  m_cellid_array                  = (Identifier*)malloc( m_ncells * sizeof( Identifier ) );

  // create an array of cell identities, they are in order of hashids.
  int ii = 0;
  for ( t_cellmap::iterator ic = m_cells->begin(); ic != m_cells->end(); ++ic ) {
     cells_Host[ii]     = *( *ic ).second;
     Identifier id      = ( ( *ic ).second )->identify();
     m_cellid_array[ii] = id;  
     ii++;  
  }
  
  ////////////////////////////////
  // omp_target_memcpy returns zero on success and nonzero on failure.
  ////////////////////////////////
  if ( omp_target_memcpy( &m_cells_d[0], cells_Host, sizeof( CaloDetDescrElement ) * m_ncells, 
                                                m_offset, m_offset, m_default_device, m_initial_device  ) ) {
    std::cout << " ERROR: Unable to copy to device." << std::endl;
    return false;
  }
  else {
    std::cout << "Target device memcpy " << ii << "/" << m_ncells << " cells"
            << " Total:" << ii * sizeof( CaloDetDescrElement ) << " Bytes" << std::endl;
  }

  free( cells_Host );

//  if ( 0 ) {
//    if ( !SanityCheck() ) { return false; }
//  }

  Rg_Sample_Index* SampleIndex_g;
  SampleIndex_g = (Rg_Sample_Index *) omp_target_alloc( sizeof( Rg_Sample_Index ) * m_ncells, m_default_device); 
  if ( SampleIndex_g == NULL ) {
    std::cout << " ERROR: No space left on device." << std::endl;;
    return false;
  }

  // copy sample_index array  to gpu
  if ( omp_target_memcpy( SampleIndex_g, m_sample_index_h, sizeof( Rg_Sample_Index ) * m_max_sample, 
                                                m_offset, m_offset, m_default_device, m_initial_device ) ) { 
     std::cout << "ERROR: copy sample index. " << std::endl;
     return false;
  }  

  // each Region allocate a grid (long Long) gpu array
  //  copy array to GPU
  //  save to regions m_cell_g ;
  for ( unsigned int ir = 0; ir < m_nregions; ++ir ) {
    //	std::cout << "debug m_regions_d[ir].cell_grid()[0] " << m_regions[ir].cell_grid()[0] <<std::endl;
    long long* ptr_g;
    ptr_g = (long long *) omp_target_alloc( sizeof( long long ) * m_regions[ir].cell_grid_eta() *
                                                        m_regions[ir].cell_grid_phi(), m_default_device); 
    if ( ptr_g == NULL ) {
      std::cout << " ERROR: No space left on device." << std::endl;;
      return false;
    }

    //      std::cout<< "cuMalloc region grid "<<  ir  << std::endl;
    if ( omp_target_memcpy( ptr_g, m_regions[ir].cell_grid(), sizeof( long long ) * m_regions[ir].cell_grid_eta() 
                              * m_regions[ir].cell_grid_phi(), m_offset, m_offset, m_default_device, m_initial_device ) ) { 
      std::cout << "ERROR: copy m_regions. " << std::endl;
      return false;
    }  
  
    //      std::cout<< "cpy grid "<<  ir  << std::endl;
    m_regions[ir].set_cell_grid_g( ptr_g );
    m_regions[ir].set_all_cells( m_cells_d ); // set this so all region instance know where the GPU cells are, before
                                              // copy to GPU
    //	std::cout<<"Gpu cell Pintor in region: " <<m_cells_d << " m_regions[ir].all_cells() " <<
    // m_regions[ir].all_cells() << std::endl ;
  }

  // GPU allocate Regions data  and load them to GPU as array of regions
  m_regions_d = (GeoRegion *) omp_target_alloc( sizeof( GeoRegion ) * m_nregions, m_default_device); 
  if ( m_regions_d == NULL ) {
    std::cout << " ERROR: No space left on device." << std::endl;;
    return false;
  }
  if ( omp_target_memcpy( m_regions_d, m_regions, sizeof( GeoRegion ) * m_nregions,
                                    m_offset, m_offset, m_default_device, m_initial_device ) ) { 
    std::cout << "ERROR: copy m_regions. " << std::endl;
    return false;
  }

//        std::cout<< "Regions Array Copied , size (Byte) " <<  sizeof(GeoRegion)*m_nregions << "sizeof cell *" <<
//        sizeof(CaloDetDescrElement *) << std::endl; std::cout<< "Region Pointer GPU print from host" <<  m_regions_d
//        << std::endl;

  GeoGpu geo_gpu_h;
  geo_gpu_h.cells        = m_cells_d;
  geo_gpu_h.ncells       = m_ncells;
  geo_gpu_h.nregions     = m_nregions;
  geo_gpu_h.regions      = m_regions_d;
  geo_gpu_h.max_sample   = m_max_sample;
  geo_gpu_h.sample_index = SampleIndex_g;

  // Now copy this to GPU and set the static member to this pointer
  GeoGpu* Gptr;
  Gptr = (GeoGpu *) omp_target_alloc( sizeof( GeoGpu ), m_default_device); 
  if ( Gptr == NULL ) {
    std::cout << " ERROR: No space left on device." << std::endl;
    return false;
  }
  if ( omp_target_memcpy( Gptr, &geo_gpu_h, sizeof( GeoGpu ),
               m_offset, m_offset, m_default_device, m_initial_device ) ) { 
    std::cout << "ERROR: copy Gptr. " << std::endl;
    return false;
  } 

  //  Geo_g = Gptr;
  m_geo_d = Gptr;

  std::cout << "\n\n m_cells_d " << m_cells_d << " m_regions_d " << m_regions_d
	  << " SampleIngex_g " << SampleIndex_g <<  " \n\n" << std::endl;
  // more test for region grids
  if ( 0 ) { return TestGeo(); }

  return true;
}

bool GeoLoadGpu::UnloadGpu_omp() {
  /**
  * Delete the memory allocated on the default device
  * during LoadGpu_omp 
  */

  omp_target_free ( m_cells_d, m_default_device ); 

  //omp_target_free ( SampleIndex_g, m_default_device ); 
  //
  //// each Region allocate a grid (long Long) gpu array
  ////  copy array to GPU
  ////  save to regions m_cell_g ;
  //for ( unsigned int ir = 0; ir < m_nregions; ++ir ) {
  //  //	std::cout << "debug m_regions_d[ir].cell_grid()[0] " << m_regions[ir].cell_grid()[0] <<std::endl;
  //  long long* ptr_g;
  //  ptr_g = (long long *) omp_target_alloc( sizeof( long long ) * m_regions[ir].cell_grid_eta() *
  //                                                      m_regions[ir].cell_grid_phi(), m_default_device); 
  //  if ( ptr_g == NULL ) {
  //    std::cout << " ERROR: No space left on device." << std::endl;;
  //    return false;
  //  }

  //  //      std::cout<< "cuMalloc region grid "<<  ir  << std::endl;
  //  if ( omp_target_memcpy( ptr_g, m_regions[ir].cell_grid(), sizeof( long long ) * m_regions[ir].cell_grid_eta() 
  //                            * m_regions[ir].cell_grid_phi(), m_offset, m_offset, m_default_device, m_initial_device ) ) { 
  //    std::cout << "ERROR: copy m_regions. " << std::endl;
  //    return false;
  //  }  
  //  //      std::cout<< "cpy grid "<<  ir  << std::endl;
  //  m_regions[ir].set_cell_grid_g( ptr_g );
  //  m_regions[ir].set_all_cells( m_cells_d ); // set this so all region instance know where the GPU cells are, before
  //                                            // copy to GPU
  //  //	std::cout<<"Gpu cell Pintor in region: " <<m_cells_d << " m_regions[ir].all_cells() " <<
  //  // m_regions[ir].all_cells() << std::endl ;
  //}

  omp_target_free ( m_regions_d, m_default_device ); 

  omp_target_free ( m_geo_d, m_default_device );

  return true;
}


