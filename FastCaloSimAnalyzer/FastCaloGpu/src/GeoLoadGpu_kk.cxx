/*
  Copyright (C) 2002-2021 CERN for the benefit of the ATLAS collaboration
*/

#include "GeoLoadGpu.h"

// needed for gpu sanity checks which are cuda kernels
// #include "GeoLoadGpu.cu"

#include <Kokkos_Core.hpp>
#include <Kokkos_Random.hpp>

GeoGpu*       GeoLoadGpu::Geo_g;
unsigned long GeoLoadGpu::num_cells;

bool GeoLoadGpu::LoadGpu_kk() {

  if ( !m_cells || m_ncells == 0 ) {
    std::cout << "Geometry is empty " << std::endl;
    return false;
  }

  std::cout << "Executing on Kokkos: " << Kokkos::DefaultExecutionSpace().name()
            << " device ";
  std::string devname{"UNKNOWN"};
#ifdef KOKKOS_ENABLE_CUDA
  cudaDeviceProp prop;
  cudaGetDeviceProperties( &prop, 0 );
  devname = prop.name;
#elif defined KOKKOS_ENABLE_HIP
  hipDeviceProp_t prop;
  auto err = hipGetDeviceProperties( &prop, 0 );
  devname = prop.name;
  if (devname == "") {
    devname = "AMD " + std::to_string(prop.gcnArch);
  }
#endif
  std::cout << devname << std::endl;


  num_cells = m_ncells;

  // Allocate Device memory for cells and copy cells as array
  // move cells on host to a array first
  m_cells_vd = Kokkos::View<CaloDetDescrElement*>( "cells", m_ncells );

  Kokkos::View<CaloDetDescrElement*>::HostMirror cells_hv = Kokkos::create_mirror_view( m_cells_vd );

  m_cellid_array = (Identifier*)malloc( m_ncells * sizeof( Identifier ) );

  // create an array of cell identities, they are in order of hashids.
  int ii = 0;
  for ( t_cellmap::iterator ic = m_cells->begin(); ic != m_cells->end(); ++ic ) {
    // cells_Host[ii]     = *( *ic ).second;
    cells_hv( ii )     = *( *ic ).second;
    Identifier id      = ( ( *ic ).second )->identify();
    m_cellid_array[ii] = id;
    ii++;
  }

  std::cout << "device Memcpy " << ii << "/" << m_ncells << " cells"
            << " Total:" << ii * sizeof( CaloDetDescrElement ) << " Bytes" << std::endl;

  Kokkos::deep_copy( m_cells_vd, cells_hv );

  m_cells_d = m_cells_vd.data();

  // if ( 0 ) {
  //   if ( !SanityCheck() ) { return false; }
  // }

  // copy sample_index array  to gpu
  m_sample_index_vd = Kokkos::View<Rg_Sample_Index*>( "sample index", m_max_sample );
  Kokkos::View<Rg_Sample_Index*, Kokkos::HostSpace> hostSampleIndex( m_sample_index_h, m_max_sample );
  Kokkos::deep_copy( m_sample_index_vd, hostSampleIndex );

  // each Region allocate a grid (long Long) gpu array
  //  copy array to GPU
  //  save to regions m_cell_g ;
  for ( unsigned int ir = 0; ir < m_nregions; ++ir ) {
    Kokkos::View<long long*>* ptr_gv =
        new Kokkos::View<long long*>( "region grid", m_regions[ir].cell_grid_eta() * m_regions[ir].cell_grid_phi() );
    m_reg_vec.push_back( ptr_gv ); // keep track to delete at end

    // create unmanged view of host ptr, copy to device
    Kokkos::View<long long*, Kokkos::HostSpace> ptr_h( m_regions[ir].cell_grid(),
                                                       m_regions[ir].cell_grid_eta() * m_regions[ir].cell_grid_phi() );
    Kokkos::deep_copy( *ptr_gv, ptr_h );

    //      std::cout<< "cpy grid "<<  ir  << std::endl;
    m_regions[ir].set_cell_grid_g( ptr_gv->data() );
    m_regions[ir].set_all_cells( m_cells_d ); // set this so all region instance know where the GPU cells are, before
                                              // copy to GPU
    //	std::cout<<"Gpu cell Pintor in region: " <<m_cells_d << " m_regions[ir].all_cells() " <<
    // m_regions[ir].all_cells() << std::endl ;
  }

  // GPU allocate Regions data  and load them to GPU as array of regions

  m_regions_vd = Kokkos::View<GeoRegion*>( "regions", m_nregions );
  Kokkos::View<GeoRegion*, Kokkos::HostSpace> reg_hv( m_regions, m_nregions );
  Kokkos::deep_copy( m_regions_vd, reg_hv );
  m_regions_d = m_regions_vd.data();

  // maybe use a DualView?
  Kokkos::View<GeoGpu>::HostMirror geo_gpu_hv = Kokkos::create_mirror_view( m_gptr_v );

  geo_gpu_hv().cells        = m_cells_d;
  geo_gpu_hv().ncells       = m_ncells;
  geo_gpu_hv().nregions     = m_nregions;
  geo_gpu_hv().regions      = m_regions_d;
  geo_gpu_hv().max_sample   = m_max_sample;
  geo_gpu_hv().sample_index = m_sample_index_vd.data();

  // Now copy this to GPU and set the member pointer
  Kokkos::deep_copy( m_gptr_v, geo_gpu_hv );
  m_geo_d = m_gptr_v.data();

  Geo_g = m_geo_d;
  
  // more test for region grids
  //  if ( 0 ) { return TestGeo(); }
  return true;
}
