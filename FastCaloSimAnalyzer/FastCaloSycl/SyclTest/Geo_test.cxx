// Copyright (C) 2002-2020 CERN for the benefit of the ATLAS collaboration

#ifdef NDEBUG
#undef NDEBUG
#endif

#include <SyclGeo/Geo.h>
#include <assert.h>

#include <CL/sycl.hpp>
#include <iostream>

#include "CaloGeometryFromFile.h"
#include "TFCSSampleDiscovery.h"

CaloGeometryFromFile* GetCaloGeo() {
  CaloGeometryFromFile* geo = new CaloGeometryFromFile();

#ifdef FCS_INPUT_PATH
  // Should supply -DINPUT_PATH if no eos access
  static const std::string fcs_input_path(FCS_INPUT_PATH);
#else
  // Must have eos access!
  static const std::string fcs_input_path(
      "/eos/atlas/atlascerngroupdisk/proj-simul");
#endif
  std::string geo_path_fcal1 =
      fcs_input_path + "/CaloGeometry/FCal1-electrodes.sorted.HV.09Nov2007.dat";
  std::string geo_path_fcal2 =
      fcs_input_path + "/CaloGeometry/FCal2-electrodes.sorted.HV.April2011.dat";
  std::string geo_path_fcal3 =
      fcs_input_path + "/CaloGeometry/FCal3-electrodes.sorted.HV.09Nov2007.dat";
  bool geo_loaded = geo->LoadGeometryFromFile(
      fcs_input_path + "/CaloGeometry/Geometry-ATLAS-R2-2016-01-00-01.root",
      TFCSSampleDiscovery::geometryTree(),
      fcs_input_path + "/CaloGeometry/cellId_vs_cellHashId_map.txt");
  if (!geo_loaded) {
    std::cout << "CaloGeometryFromFile::LoadGeometryFromFile() failed!\n";
    return nullptr;
  }
  geo->LoadFCalGeometryFromFiles(geo_path_fcal1, geo_path_fcal2,
                                 geo_path_fcal3);
  return geo;
}

class RegionKernel {
 public:
  RegionKernel() = delete;
  RegionKernel(DeviceGeo* devgeo) : devgeo_(devgeo) {}
  void operator()(cl::sycl::id<1> idx) {
    unsigned int id = (int)idx[0];
    unsigned int neta = devgeo_->regions[id].cell_grid_eta();
    unsigned int nphi = devgeo_->regions[id].cell_grid_phi();
    cl::sycl::intel::experimental::printf(
        fastcalosycl::syclcommon::kPrintNEtaNPhi, neta, nphi);
  }

 private:
  DeviceGeo* devgeo_;
};

// Test device-side cells.
void test_device_cells(cl::sycl::queue* q, const DeviceGeo* dev_geo,
                       const unsigned long ncells) {
// CUDA does not support experimental::printf()
#ifndef SYCL_TARGET_CUDA
  std::cout << "Test device cells..." << std::endl;
  auto ev_cellinfo = q->submit([&](cl::sycl::handler& cgh) {
    cgh.parallel_for<class Dummy>(
        cl::sycl::range<1>(ncells), [=](cl::sycl::id<1> idx) {
          // cl::sycl::range<1>(ncells),
          // [=, dev_cells_local = dev_geo->cells](cl::sycl::id<1> idx) {
          unsigned int id = (int)idx[0];
          if ((id + 1) % 10000 == 0) {
            long long cell_id = dev_geo->cells[id].identify();
            unsigned long long hash = dev_geo->cells[id].calo_hash();
            cl::sycl::intel::experimental::printf(
                fastcalosycl::syclcommon::kCellInfo, cell_id, hash);
          }
        });
  });
  ev_cellinfo.wait_and_throw();
#else
  std::cout << "CUDA does not support experimental::printf(). Cannot call "
               "test_cells()."
            << std::endl;
#endif
}

// Test regions.
void test_regions(cl::sycl::queue* q, const DeviceGeo* dev_geo) {
  // CUDA does not support experimental::printf()
#ifndef SYCL_TARGET_CUDA
  std::cout << "Test device region..." << std::endl;
  auto ev_region_info = q->submit([&](cl::sycl::handler& cgh) {
    cgh.parallel_for<class RegionInfo>(
        cl::sycl::range<1>(dev_geo->num_regions), [=](cl::sycl::id<1> idx) {
          unsigned int id = (int)idx[0];
          unsigned int neta = dev_geo->regions[id].cell_grid_eta();
          unsigned int nphi = dev_geo->regions[id].cell_grid_phi();
          cl::sycl::intel::experimental::printf(
              fastcalosycl::syclcommon::kPrintNEtaNPhi, neta, nphi);
        });
  });
  ev_region_info.wait_and_throw();
#else
  std::cout << "CUDA does not support experimental::printf(). Cannot call "
               "test_regions()."
            << std::endl;
#endif
}

// Copy device cell data back to host.
// void copy_device_geo_to_host(cl::sycl::queue* q, DeviceGeo* dev_geo,
//                              DeviceGeo* host_geo) {
//   std::cout << "Copying device cells to host... " << std::endl;
//   // Start timer.
//   auto geo_cpy_start2 = std::chrono::system_clock::now();
//   queue
//       .memcpy(&host_cells_[0], device_cells_,
//               num_cells_ * sizeof(CaloDetDescrElement))
//       .wait_and_throw();
//   // End timer.
//   auto geo_cpy_end2 = std::chrono::system_clock::now();
//   // Time to copy geometry host->device.
//   auto geo_cpy_dur2 =
//       std::chrono::duration<double>(geo_cpy_end2 - geo_cpy_start2);
//   std::cout << "\tCells copied in " << std::dec << geo_cpy_dur2.count()
//             << " ms.\n"
//             << std::endl;
// }

// main
int main() {
  std::cout << "*** Geo_test BEGINS ***" << std::endl;

  // Initialize a Geo object
  Geo* geo = new Geo();
  CaloGeometryFromFile* calo_geo = GetCaloGeo();
  if (!calo_geo) {
    std::cout << "Geo_test::GetCaloGeo() failed!\n";
    return -1;
  }
  geo->Init(calo_geo);

  // Load geometry to device.
  bool load_success = geo->LoadDeviceGeo();
  if (!load_success) {
    std::cout << "Could not load device geometry!" << std::endl;
    return -1;
  }

  // Get the DeviceGeo and queue from the Geo object.
  DeviceGeo* device_geo = Geo::DevGeo;
  cl::sycl::queue device_q = geo->GetGeoDeviceQueue();

  // First print select host-side cells.
  const CaloDetDescrElement* host_cells = geo->GetCells();
  const unsigned long num_cells = geo->GetNumCells();
  std::cout << "Test host cells..." << std::endl;
  for (unsigned long icell = 0; icell < num_cells; ++icell) {
    if ((icell + 1) % 10000 == 0) {
      std::cout << "  host_cell :: id [" << std::hex
                << host_cells[icell].identify() << std::dec << "], hash_id ["
                << host_cells[icell].calo_hash() << "]" << std::endl;
    }
  }

  // Test device-side cells, c.f. host-side cells.
  test_device_cells(&device_q, device_geo, num_cells);
  // test_regions(&device_q, device_geo);
  std::cout << "Test device region..." << std::endl;
  device_q
      .submit([&](cl::sycl::handler& cgh) {
        RegionKernel kernel(device_geo);
        cgh.parallel_for(geo->GetNumRegions(), kernel);
      })
      .wait_and_throw();

  // Test specific cell
  const CaloDetDescrElement* test_cell = 0;
  unsigned long long cellid64(3179554531063103488);
  Identifier cellid(cellid64);
  test_cell = calo_geo->getDDE(cellid);
  if (!test_cell) {
    std::cout << "Test cell not found!\n";
    return -1;
  }
  std::cout << "\nTest cell:" << std::endl
            << "\tIdentifier: " << test_cell->identify() << std::endl
            << "\tSampling: " << test_cell->getSampling() << std::endl
            << "\teta: " << test_cell->eta() << std::endl
            << "\tphi: " << test_cell->phi() << std::endl
            << "\tCaloDetDescrElement: " << test_cell << std::endl;

  const CaloDetDescrElement* test_cell_host = geo->FindCell(cellid);
  if (!test_cell_host) {
    std::cout << "Test cell (host) not found!\n";
    return -1;
  }
  std::cout << "\nTest cell (host):" << std::endl
            << "\tIdentifier: " << test_cell_host->identify() << std::endl
            << "\tSampling: " << test_cell_host->getSampling() << std::endl
            << "\teta: " << test_cell_host->eta() << std::endl
            << "\tphi: " << test_cell_host->phi() << std::endl
            << "\tCaloDetDescrElement: " << test_cell_host << std::endl;

  std::cout << "\n*** Geo_test ENDS ***" << std::endl;
  return 0;
}  // main