/*
  Copyright (C) 2002-2021 CERN for the benefit of the ATLAS collaboration
*/

#include "GeoLoadGpu.h"
#include "AlpakaDefs.h"
#include <alpaka/alpaka.hpp>
#include <iostream>
#include "gpuQ.h"

class GeoLoadGpu::Impl {
public:
  Impl()
    : sample_index_buf_host{alpaka::allocBuf<Rg_Sample_Index, Idx>(alpaka::getDevByIdx<Host>(0u), Idx{1})}
    , regions_buf_host{alpaka::allocBuf<GeoRegion, Idx>(alpaka::getDevByIdx<Host>(0u), Idx{1})}
    , geogpu_buf_host{alpaka::allocBuf<GeoGpu, Idx>(alpaka::getDevByIdx<Host>(0u), Idx{1})}
    , cells_buf_acc{alpaka::allocBuf<CaloDetDescrElement, Idx>(alpaka::getDevByIdx<Acc>(0u), Idx{1})}
    , sample_index_buf_acc{alpaka::allocBuf<Rg_Sample_Index, Idx>(alpaka::getDevByIdx<Acc>(0u), Idx{1})}
    , regions_buf_acc{alpaka::allocBuf<GeoRegion, Idx>(alpaka::getDevByIdx<Acc>(0u), Idx{1})}
    , geogpu_buf_acc{alpaka::allocBuf<GeoGpu, Idx>(alpaka::getDevByIdx<Acc>(0u), Idx{1})}
  {
  }

  BufHostSampleIndex sample_index_buf_host;
  BufHostGeoRegion regions_buf_host;
  BufHostGeoGpu geogpu_buf_host;
  std::vector<BufHostLongLong> reg_vec_host;

  BufAccCaloDDE cells_buf_acc;
  BufAccSampleIndex sample_index_buf_acc;
  BufAccGeoRegion regions_buf_acc;
  BufAccGeoGpu geogpu_buf_acc;
  std::vector<BufAccLongLong> reg_vec_acc;
};

/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/* Test Kernels */
struct TestHelloKernel
{
  template<typename TAcc>
  ALPAKA_FN_ACC auto operator()(TAcc const& acc) const -> void
  {
    auto const idx = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[0];
    printf( "Hello, I am from GPU thread %d\n", static_cast<unsigned>(idx) );
  }
};

struct TestCellKernel
{
  template<typename TAcc>
  ALPAKA_FN_ACC auto operator()(TAcc const& acc
				, const CaloDetDescrElement* cells
				, unsigned long index) const -> void
  {
    const CaloDetDescrElement* cell   = &cells[index];
    int                  sample = cell->getSampling();
    float                eta    = cell->eta();
    float                phi    = cell->phi();

    long long hashid = cell->calo_hash();

    printf( "From GPU cell index %ld , hashid=%ld, eta=%f, phi=%f, sample=%d \n"
	    , index
	    , hashid
	    , eta
	    , phi
	    , sample );
  }
};

struct TestGeoKernel
{
  template<typename TAcc>
  ALPAKA_FN_ACC auto operator()(TAcc const& acc
				, const CaloDetDescrElement* cells
				, const GeoRegion* regions
				, unsigned int nregions
				, unsigned long ncells
				, int r, int ir, int ip) const -> void
  {
    int                neta  = regions[r].cell_grid_eta();
    int                nphi  = regions[r].cell_grid_phi();
    unsigned long long index = regions[r].cell_grid_g()[ir * nphi + ip];
    printf( " From GPU.., region %d, cell_grid[%d][%d]: [%d][%d] index=%lu \n"
	    , r
	    , ir
	    , ip
	    , neta
	    , nphi
	    , index );

  const CaloDetDescrElement* c = &cells[index];

  long long hashid = c->calo_hash();
  long long id     = c->identify();
  int       sample = c->getSampling();
  float     eta    = c->eta();
  float     phi    = c->phi();

  printf( " From GPU.., region %d, cell_grid[%d][%d]: index %lu index, hashid=%ld,eta=%f, phi=%f, sample=%d , ID=%ld "
          "cell_ptr=%#015lx \n"
	  , r
	  , ir
	  , ip
	  , index
	  , hashid
	  , eta
	  , phi
	  , sample
	  , id
	  , regions[r].all_cells() );
  
  CaloDetDescrElement cc = ( regions[r].all_cells() )[index];
  printf( " GPU test region have cells: cell index %lu, eta=%f phi=%f size of cell*GPU=%lu\n"
	  , index
	  , cc.eta()
	  , cc.phi()
	  , sizeof( CaloDetDescrElement* ) );
  }
};

struct TestGeoKernel_G
{
  template<typename TAcc>
  ALPAKA_FN_ACC auto operator()(TAcc const& acc
				, const GeoGpu* geo
				, int r, int ir, int ip) const -> void
  {
    const GeoRegion*           regions = ( *geo ).regions;
    const CaloDetDescrElement* cells   = geo->cells;

    int                neta  = regions[r].cell_grid_eta();
    int                nphi  = regions[r].cell_grid_phi();
    unsigned long long index = regions[r].cell_grid_g()[ir * nphi + ip];
    printf( " From GPU.., region %d, cell_grid[%d][%d]: [%d][%d] index=%ld \n"
	    , r
	    , ir
	    , ip
	    , neta
	    , nphi
	    , index );

  const CaloDetDescrElement* c = &cells[index];

  long long hashid = c->calo_hash();
  long long id     = c->identify();
  int       sample = c->getSampling();
  float     eta    = c->eta();
  float     phi    = c->phi();

  printf( " From GPU.., region %d, cell_grid[%d][%d]: index %lu index, hashid=%ld,eta=%f, phi=%f, sample=%d , ID=%ld "
          "cell_ptr=%#015lx \n"
	  , r
	  , ir
	  , ip
	  , index
	  , hashid
	  , eta
	  , phi
	  , sample
	  , id
	  , regions[r].all_cells() );

  CaloDetDescrElement cc = ( regions[r].all_cells() )[index];
  printf( " GPU test region have cells: cell index %llu, eta=%f phi=%f size of cell*GPU=%lu\n"
	  , index
	  , cc.eta()
	  , cc.phi()
	  , sizeof( CaloDetDescrElement* ) );
  }
};

/* Test Utilities */
bool GeoLoadGpu::TestGeo() {
  QueueAcc queue(alpaka::getDevByIdx<Acc>(Idx{0}));

  WorkDiv workdiv(Idx{1},Idx{1},Idx{1});
  TestGeoKernel testGeo;
  alpaka::exec<Acc>(queue
		    , workdiv
		    , testGeo
		    , m_cells_d
		    , m_regions_d
		    , m_ncells
		    , m_nregions
		    , 14, 0, 32);
  TestGeoKernel_G testGeo_G;
  alpaka::exec<Acc>(queue
		    , workdiv
		    , testGeo_G
		    , m_geo_d
		    , 14, 0, 32);
  alpaka::wait(queue);

  std::cout << "TesGeo finished " << std::endl;

  long long* c   = m_regions[14].cell_grid();
  int        np  = m_regions[14].cell_grid_phi();
  int        ne  = m_regions[14].cell_grid_eta();
  int        idx = c[0 * np + 32];
  Identifier Id  = m_cellid_array[idx];
  std::cout << "From Host: Region[14]Grid[0][32]: index=" << idx << ", ID=" << Id
            << ", HashCPU=" << ( *m_cells )[Id]->calo_hash() << ", neta=" << ne << ",  nphi=" << np
            << ", eta=" << ( *m_cells )[Id]->eta() << std::endl;

  return true;
  // end test
}
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

bool GeoLoadGpu::SanityCheck() {
  // sanity check/test
  QueueAcc queue(alpaka::getDevByIdx<Acc>(Idx{0}));

  WorkDiv workdiv(Idx{1},Idx{1},Idx{1});
  TestHelloKernel testHello;
  alpaka::exec<Acc>(queue
		    , workdiv
		    , testHello);
  TestCellKernel testCell;
  alpaka::exec<Acc>(queue
		    , workdiv
		    , testCell
		    , m_cells_d
		    , 1872);
  alpaka::wait(queue);

  std::cout << " ID of 2000's cell " << m_cellid_array[2000] << std::endl;
  Identifier Id = m_cellid_array[2000];
  std::cout << "ID of cell 2000: " << ( *m_cells )[Id]->identify() << "hashid: " << ( *m_cells )[Id]->calo_hash()
            << std::endl;
  std::cout << "Size of Identify: " << sizeof( Identifier ) << "size of Region: " << sizeof( GeoRegion ) << std::endl;

  std::cout << "GPU Kernel cell test lauched" << std::endl;

  return true;
}

/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

Rg_Sample_Index* GeoLoadGpu::get_sample_index_h_al()
{
  if(!pImpl) pImpl = new Impl();

  pImpl->sample_index_buf_host = alpaka::allocBuf<Rg_Sample_Index, Idx>(alpaka::getDevByIdx<Host>(0u), static_cast<Idx>(m_max_sample));
  return (Rg_Sample_Index*)alpaka::getPtrNative(pImpl->sample_index_buf_host);
}

GeoRegion* GeoLoadGpu::get_regions_al()
{
  if(!pImpl) pImpl = new Impl();

  pImpl->regions_buf_host = alpaka::allocBuf<GeoRegion, Idx>(alpaka::getDevByIdx<Host>(0u), Idx{m_nregions});
  return (GeoRegion*)alpaka::getPtrNative(pImpl->regions_buf_host);
}

long long* GeoLoadGpu::get_cell_grid_al(int neta, int nphi)
{
  if(!pImpl) pImpl = new Impl();

  BufHostLongLong cells = alpaka::allocBuf<long long, Idx>(alpaka::getDevByIdx<Host>(0u), static_cast<Idx>(neta*nphi));
  pImpl->reg_vec_host.push_back(cells);
  return (long long*)alpaka::getPtrNative(cells);
}

bool GeoLoadGpu::LoadGpu_al() {
  if ( !m_cells || m_ncells == 0 ) {
    std::cout << "Geometry is empty " << std::endl;
    return false;
  }

  if(!pImpl) pImpl = new Impl();

  QueueAcc queue(alpaka::getDevByIdx<Acc>(Idx{0}));
  
  BufHostGeoGpu geo_gpu_host = alpaka::allocBuf<GeoGpu,Idx>(alpaka::getDevByIdx<Host>(0u),Idx{1});
  GeoGpu* geo_gpu_ptr = alpaka::getPtrNative(geo_gpu_host);

  // Allocate Device memory for cells and copy cells as array
  // move cells on host to a array first
  pImpl->cells_buf_acc = alpaka::allocBuf<CaloDetDescrElement, Idx>(alpaka::getDevByIdx<Acc>(0u), m_ncells);
  m_cells_d = alpaka::getPtrNative(pImpl->cells_buf_acc);

  std::cout << "cuMalloc " << m_ncells << " cells" << std::endl;

  BufHostCaloDDE cells_buf_host = alpaka::allocBuf<CaloDetDescrElement, Idx>(alpaka::getDevByIdx<Host>(0u),m_ncells);
  CaloDetDescrElement* cells_Host = alpaka::getPtrNative(cells_buf_host);

  m_cellid_array                  = (Identifier*)malloc(m_ncells*sizeof(Identifier));

  // create an array of cell identities, they are in order of hashids.
  int ii = 0;
  for(t_cellmap::iterator ic = m_cells->begin(); ic != m_cells->end(); ++ic) {
    cells_Host[ii]     = *( *ic ).second;
    Identifier id      = ( ( *ic ).second )->identify();
    m_cellid_array[ii] = id;
    ii++;
  }

  alpaka::memcpy(queue,pImpl->cells_buf_acc,cells_buf_host);
  alpaka::wait(queue);
  
  std::cout << "device Memcpy " << ii << "/" << m_ncells << " cells"
            << " Total:" << ii * sizeof( CaloDetDescrElement ) << " Bytes" << std::endl;

  if ( 0 ) {
    if ( !SanityCheck() ) { return false; }
  }

  pImpl->sample_index_buf_acc = alpaka::allocBuf<Rg_Sample_Index, Idx>(alpaka::getDevByIdx<Acc>(0u), static_cast<Idx>(m_max_sample));
  Rg_Sample_Index* SampleIndex_g = alpaka::getPtrNative(pImpl->sample_index_buf_acc);

  // copy sample_index array  to gpu
  alpaka::memcpy(queue,pImpl->sample_index_buf_acc,pImpl->sample_index_buf_host);
  alpaka::wait(queue);

  // each Region allocate a grid (long Long) gpu array
  //  copy array to GPU
  //  save to regions m_cell_g ;
  for(unsigned int ir = 0; ir < m_nregions; ++ir) {
    BufAccLongLong gv = alpaka::allocBuf<long long, Idx>(alpaka::getDevByIdx<Acc>(0u), static_cast<Idx>(m_regions[ir].cell_grid_eta()* m_regions[ir].cell_grid_phi()));
    pImpl->reg_vec_acc.push_back(gv);
    long long* ptr_g = alpaka::getPtrNative(gv);

    alpaka::memcpy(queue,gv,pImpl->reg_vec_host[ir]);
    alpaka::wait(queue);

    m_regions[ir].set_cell_grid_g( ptr_g );
    m_regions[ir].set_all_cells( m_cells_d ); // set this so all region instance know where the GPU cells are, before copy to GPU
  }

  // GPU allocate Regions data  and load them to GPU as array of regions

  pImpl->regions_buf_acc = alpaka::allocBuf<GeoRegion,Idx>(alpaka::getDevByIdx<Acc>(0u), Idx{m_nregions});
  m_regions_d = alpaka::getPtrNative(pImpl->regions_buf_acc);

  alpaka::memcpy(queue,pImpl->regions_buf_acc,pImpl->regions_buf_host);

  geo_gpu_ptr->cells        = m_cells_d;
  geo_gpu_ptr->ncells       = m_ncells;
  geo_gpu_ptr->nregions     = m_nregions;
  geo_gpu_ptr->regions      = m_regions_d;
  geo_gpu_ptr->max_sample   = m_max_sample;
  geo_gpu_ptr->sample_index = SampleIndex_g;


  std::cout << "GEO\n";
  std::cout << "ncells: " << geo_gpu_ptr->ncells << "\n";
  std::cout << "regions: " << m_nregions << "\n";
  std::cout << "=================================\n";
  
  // Now copy this to GPU and set the static member to this pointer
  pImpl->geogpu_buf_acc = alpaka::allocBuf<GeoGpu,Idx>(alpaka::getDevByIdx<Acc>(0u), Idx{sizeof(GeoGpu)});
  GeoGpu* Gptr = alpaka::getPtrNative(pImpl->geogpu_buf_acc);

  alpaka::memcpy(queue,pImpl->geogpu_buf_acc,geo_gpu_host);
  alpaka::wait(queue);
  
  m_geo_d = Gptr;

  // more test for region grids
  if ( 0 ) { return TestGeo(); }
  return true;
}

