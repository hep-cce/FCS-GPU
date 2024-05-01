#include "hip/hip_runtime.h"
/*
  Copyright (C) 2002-2021 CERN for the benefit of the ATLAS collaboration
*/

#include <iostream>
#include "GeoLoadGpu.h"
#include "gpuQ.h"

__global__ void testHello() {
  printf("Hello, I am from GPU thread %zu\n", (size_t)threadIdx.x);
}

/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

__global__ void testCell(CaloDetDescrElement *cells, unsigned long index) {
  CaloDetDescrElement *cell = &cells[index];
  int sample = cell->getSampling();
  float eta = cell->eta();
  float phi = cell->phi();

  long long hashid = cell->calo_hash();

  printf(" From GPU cell index %ld , hashid=%lld, eta=%f, phi=%f, sample=%d \n",
         index, hashid, eta, phi, sample);
}

/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

__global__ void testGeo(CaloDetDescrElement *cells, GeoRegion *regions,
                        unsigned int nregions, unsigned long ncells, int r,
                        int ir, int ip) {

  int neta = regions[r].cell_grid_eta();
  int nphi = regions[r].cell_grid_phi();
  unsigned long long index = regions[r].cell_grid_g()[ir * nphi + ip];
  printf(" From GPU.., region %d, cell_grid[%d][%d]: [%d][%d] index=%llu \n", r,
         ir, ip, neta, nphi, index);

  CaloDetDescrElement *c = &cells[index];

  long long hashid = c->calo_hash();
  long long id = c->identify();
  int sample = c->getSampling();
  float eta = c->eta();
  float phi = c->phi();

  printf(" From GPU.., region %d, cell_grid[%d][%d]: index %llu index, "
         "hashid=%lld,eta=%f, phi=%f, sample=%d , ID=%lld "
         "cell_ptr=%p \n",
         r, ir, ip, index, hashid, eta, phi, sample, id,
         regions[r].all_cells());

  CaloDetDescrElement cc = (regions[r].all_cells())[index];
  printf(" GPU test region have cells: cell index %llu, eta=%f phi=%f size of "
         "cell*GPU=%lu\n",
         index, cc.eta(), cc.phi(), sizeof(CaloDetDescrElement *));
}

/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

__global__ void testGeo_g(GeoGpu *geo, int r, int ir, int ip) {

  GeoRegion *regions = (*geo).regions;
  CaloDetDescrElement *cells = geo->cells;

  int neta = regions[r].cell_grid_eta();
  int nphi = regions[r].cell_grid_phi();
  unsigned long long index = regions[r].cell_grid_g()[ir * nphi + ip];
  printf(" From GPU.., region %d, cell_grid[%d][%d]: [%d][%d] index=%llu \n", r,
         ir, ip, neta, nphi, index);

  CaloDetDescrElement *c = &cells[index];

  long long hashid = c->calo_hash();
  long long id = c->identify();
  int sample = c->getSampling();
  float eta = c->eta();
  float phi = c->phi();

  printf(" From GPU.., region %d, cell_grid[%d][%d]: index %llu index, "
         "hashid=%lld,eta=%f, phi=%f, sample=%d , ID=%lld "
         "cell_ptr=%p \n",
         r, ir, ip, index, hashid, eta, phi, sample, id,
         regions[r].all_cells());

  CaloDetDescrElement cc = (regions[r].all_cells())[index];
  printf(" GPU test region have cells: cell index %llu, eta=%f phi=%f size of "
         "cell*GPU=%lu\n",
         index, cc.eta(), cc.phi(), sizeof(CaloDetDescrElement *));
}

/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

bool GeoLoadGpu::TestGeo() {
  hipLaunchKernelGGL(testGeo, 1, 1, 0, 0, m_cells_d, m_regions_d, m_ncells, m_nregions, 14, 0, 32);
  gpuQ(hipDeviceSynchronize());
  hipError_t err = hipGetLastError();
  if (err != hipSuccess) {
    std::cout << hipGetErrorString(err) << std::endl;
    return false;
  }

  hipLaunchKernelGGL(testGeo_g, 1, 1, 0, 0, get_geoPtr(), 14, 0, 32);
  gpuQ(hipDeviceSynchronize());
  err = hipGetLastError();
  if (err != hipSuccess) {
    std::cout << hipGetErrorString(err) << std::endl;
    return false;
  }

  std::cout << "TesGeo finished " << std::endl;

  long long *c = m_regions[14].cell_grid();
  int np = m_regions[14].cell_grid_phi();
  int ne = m_regions[14].cell_grid_eta();
  int idx = c[0 * np + 32];
  Identifier Id = m_cellid_array[idx];
  std::cout << "From Host: Region[14]Grid[0][32]: index=" << idx
            << ", ID=" << Id << ", HashCPU=" << (*m_cells)[Id]->calo_hash()
            << ", neta=" << ne << ",  nphi=" << np
            << ", eta=" << (*m_cells)[Id]->eta() << std::endl;

  return true;
  // end test
}

/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
bool GeoLoadGpu::SanityCheck() {
  // sanity check/test
  hipLaunchKernelGGL(testHello, 1, 1, 0, 0);
  hipLaunchKernelGGL(testCell, 1, 1, 0, 0, m_cells_d, 1872);
  gpuQ(hipDeviceSynchronize());

  std::cout << " ID of 2000's cell " << m_cellid_array[2000] << std::endl;
  Identifier Id = m_cellid_array[2000];
  std::cout << "ID of cell 2000: " << (*m_cells)[Id]->identify()
            << "hashid: " << (*m_cells)[Id]->calo_hash() << std::endl;
  std::cout << "Size of Identify: " << sizeof(Identifier)
            << "size of Region: " << sizeof(GeoRegion) << std::endl;

  std::cout << "GPU Kernel cell test lauched" << std::endl;

  hipError_t err = hipGetLastError();
  if (err != hipSuccess) {
    std::cout << hipGetErrorString(err) << std::endl;
    return false;
  }
  return true;
}

/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

GeoGpu *GeoLoadGpu::Geo_g;
unsigned long GeoLoadGpu::num_cells;

bool GeoLoadGpu::LoadGpu_cu() {
  if (!m_cells || m_ncells == 0) {
    std::cout << "Geometry is empty " << std::endl;
    return false;
  }

  hipDeviceProp_t prop;
  gpuQ(hipGetDeviceProperties(&prop, 0));
  std::cout << "Executing on GPU: " << prop.name << std::endl;

  GeoGpu geo_gpu_h;
  num_cells = m_ncells;

  // Allocate Device memory for cells and copy cells as array
  // move cells on host to a array first
  if (hipSuccess !=
      hipMalloc((void **)&m_cells_d, sizeof(CaloDetDescrElement) * m_ncells))
    return false;
  std::cout << "cuMalloc " << m_ncells << " cells" << std::endl;

  CaloDetDescrElement *cells_Host =
      (CaloDetDescrElement *)malloc(m_ncells * sizeof(CaloDetDescrElement));
  m_cellid_array = (Identifier *)malloc(m_ncells * sizeof(Identifier));

  // create an array of cell identities, they are in order of hashids.
  int ii = 0;
  for (t_cellmap::iterator ic = m_cells->begin(); ic != m_cells->end(); ++ic) {
    cells_Host[ii] = *(*ic).second;
    Identifier id = ((*ic).second)->identify();
    m_cellid_array[ii] = id;
    ii++;
  }

  if (hipSuccess != hipMemcpy(&m_cells_d[0], cells_Host,
                                sizeof(CaloDetDescrElement) * m_ncells,
                                hipMemcpyHostToDevice))
    return false;
  std::cout << "device Memcpy " << ii << "/" << m_ncells << " cells"
            << " Total:" << ii * sizeof(CaloDetDescrElement) << " Bytes"
            << std::endl;

  free(cells_Host);

  if (0) {
    if (!SanityCheck()) {
      return false;
    }
  }

  Rg_Sample_Index *SampleIndex_g;
  if (hipSuccess != hipMalloc((void **)&SampleIndex_g,
                                sizeof(Rg_Sample_Index) * m_max_sample))
    return false;

  // copy sample_index array  to gpu
  if (hipSuccess != hipMemcpy(SampleIndex_g, m_sample_index_h,
                                sizeof(Rg_Sample_Index) * m_max_sample,
                                hipMemcpyHostToDevice)) {
    std::cout << "Error copy sample index " << std::endl;

    return false;
  }

  // each Region allocate a grid (long Long) gpu array
  //  copy array to GPU
  //  save to regions m_cell_g ;
  for (unsigned int ir = 0; ir < m_nregions; ++ir) {
    //	std::cout << "debug m_regions_d[ir].cell_grid()[0] " <<
    //m_regions[ir].cell_grid()[0] <<std::endl;
    long long *ptr_g;
    if (hipSuccess !=
        hipMalloc((void **)&ptr_g, sizeof(long long) *
                                        m_regions[ir].cell_grid_eta() *
                                        m_regions[ir].cell_grid_phi()))
      return false;
    //      std::cout<< "cuMalloc region grid "<<  ir  << std::endl;
    if (hipSuccess !=
        hipMemcpy(ptr_g, m_regions[ir].cell_grid(),
                   sizeof(long long) * m_regions[ir].cell_grid_eta() *
                       m_regions[ir].cell_grid_phi(),
                   hipMemcpyHostToDevice))
      return false;
    //      std::cout<< "cpy grid "<<  ir  << std::endl;
    m_regions[ir].set_cell_grid_g(ptr_g);
    m_regions[ir].set_all_cells(m_cells_d); // set this so all region instance
                                            // know where the GPU cells are,
                                            // before
                                            // copy to GPU
    //	std::cout<<"Gpu cell Pintor in region: " <<m_cells_d << "
    //m_regions[ir].all_cells() " <<
    // m_regions[ir].all_cells() << std::endl ;
  }

  // GPU allocate Regions data  and load them to GPU as array of regions

  if (hipSuccess !=
      hipMalloc((void **)&m_regions_d, sizeof(GeoRegion) * m_nregions))
    return false;
  //      std::cout<< "cuMalloc "<< m_nregions << " regions" << std::endl;
  if (hipSuccess != hipMemcpy(m_regions_d, m_regions,
                                sizeof(GeoRegion) * m_nregions,
                                hipMemcpyHostToDevice))
    return false;
  //        std::cout<< "Regions Array Copied , size (Byte) " <<
  // sizeof(GeoRegion)*m_nregions << "sizeof cell *" <<
  //        sizeof(CaloDetDescrElement *) << std::endl; std::cout<< "Region
  // Pointer GPU print from host" <<  m_regions_d
  //        << std::endl;

  geo_gpu_h.cells = m_cells_d;
  geo_gpu_h.ncells = m_ncells;
  geo_gpu_h.nregions = m_nregions;
  geo_gpu_h.regions = m_regions_d;
  geo_gpu_h.max_sample = m_max_sample;
  geo_gpu_h.sample_index = SampleIndex_g;

  std::cout << "ncells: " << geo_gpu_h.ncells << "\n";
  std::cout << "regions: " << m_nregions << "\n";

  // Now copy this to GPU and set the static member to this pointer
  GeoGpu *Gptr;
  gpuQ(hipMalloc((void **)&Gptr, sizeof(GeoGpu)));
  gpuQ(hipMemcpy(Gptr, &geo_gpu_h, sizeof(GeoGpu), hipMemcpyHostToDevice));

  Geo_g = Gptr;
  m_geo_d = Gptr;

  // more test for region grids
  if (0) {
    return TestGeo();
  }
  return true;
}
