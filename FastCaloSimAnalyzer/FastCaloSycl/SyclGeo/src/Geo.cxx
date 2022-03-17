// Copyright (C) 2002-2020 CERN for the benefit of the ATLAS collaboration

#include <SyclGeo/Geo.h>

#include <chrono>

// Geo
Geo::Geo()
    : num_cells_(0UL),
      num_regions_(0),
      cell_map_(nullptr),
      regions_(nullptr),
      regions_device_(nullptr),
      cells_device_(nullptr),
      cell_id_(nullptr),
      max_sample_(0U),
      sample_index_(nullptr),
      ctx_(nullptr) {}

Geo::Geo(cl::sycl::context* ctx)
    : num_cells_(0UL),
      num_regions_(0),
      cell_map_(nullptr),
      regions_(nullptr),
      regions_device_(nullptr),
      cells_device_(nullptr),
      cell_id_(nullptr),
      max_sample_(0U),
      sample_index_(nullptr),
      ctx_(ctx) {}

Geo::~Geo() {
  free(cells_);
  cells_ = nullptr;
  free(regions_);
  regions_ = nullptr;
  free(cell_id_);
  cell_id_ = nullptr;
  free(cell_map_);
  cell_map_ = nullptr;
  cl::sycl::free(&cells_device_, *ctx_);
  cl::sycl::free(&regions_device_, *ctx_);
  // cl::sycl::free(&device_geo_, ctx_);
}

bool Geo::AllocMemCells(cl::sycl::device* dev) {
  // Allocate host-side memory for cell array.
  cells_ =
      (CaloDetDescrElement*)malloc(num_cells_ * sizeof(CaloDetDescrElement));
  cell_id_ = (Identifier*)malloc(num_cells_ * sizeof(Identifier));

  if (!cells_ || !cell_id_) {
    std::cout << "Cannot allocate host-side memory!" << std::endl;
    return false;
  }

  // Allocate device-side memory for cell array.
  cells_device_ = (CaloDetDescrElement*)malloc_device(
      num_cells_ * sizeof(CaloDetDescrElement), (*dev), *ctx_);
  if (!cells_device_) {
    std::cout << "Cannot allocate device-side memory!" << std::endl;
    return false;
  }

  // Memory allocated successfully.
  std::cout << "Host and device cell memory allocated..." << std::endl;
  std::cout << "\tnum_cells: " << num_cells_ << std::endl
            << "\tsize: "
            << (int)num_cells_ * sizeof(CaloDetDescrElement) / 1000 << " kb\n"
            << std::endl;
  return true;
}  // AllocMemCells

bool Geo::Init(CaloGeometryFromFile* cg) {
  // Allocate host memory for geometry regions
  unsigned int num_regions = cg->get_tot_regions();
  GeoRegion* geo_regions = (GeoRegion*)malloc(num_regions * sizeof(GeoRegion));
  if (!geo_regions) {
    std::cout << "FastCaloSycl::Geo\tCould not allocate GeoRegion memory!";
    return false;
  }

  // Allocate host memory for geometry samples
  sample_index_ =
      (SampleIndex*)malloc(CaloGeometry::MAX_SAMPLING * sizeof(SampleIndex));
  if (!sample_index_) {
    std::cout << "FastCaloSycl::Geo\tCould not allocate SampleIndex memory!";
    return false;
  }

  // Prepare device geometry.
  // Analogous to TFCSShapeValidation::GeoL()
  this->set_num_cells(cg->get_cells()->size());
  this->set_max_sample(CaloGeometry::MAX_SAMPLING);
  this->set_num_regions(num_regions);
  this->set_cell_map(cg->get_cells());
  this->set_host_regions(geo_regions);
  this->set_sample_index(sample_index_);

  // Copy geometry from CaloGeometryFromFile
  // Loop over samples
  unsigned int i = 0;
  for (int isamp = 0; isamp < CaloGeometry::MAX_SAMPLING; ++isamp) {
    // Get the number of regions in this sampling
    unsigned int num_regions = cg->get_n_regions(isamp);

    // Assign sampling information
    sample_index_[isamp].index = i;
    sample_index_[isamp].size = num_regions;
    // Loop over regions
    for (unsigned int ireg = 0; ireg < num_regions; ++ireg) {
      this->CopyGeoRegion(cg->get_region(isamp, ireg), &geo_regions[i++]);
    }  // Loop over regions
  }    // Loop over samples
  return true;
}

DeviceGeo* Geo::DevGeo;
bool Geo::LoadDeviceGeo() {
  if (!cell_map_ || num_cells_ == 0) {
    std::cout << "Empty geometry!" << std::endl;
    return false;
  }

  // Catch asynchronous exceptions
  auto exception_handler = [](cl::sycl::exception_list exceptions) {
    for (std::exception_ptr const& e : exceptions) {
      try {
        std::rethrow_exception(e);
      } catch (cl::sycl::exception const& e) {
        std::cout << "Caught asynchronous SYCL exception during generation:\n"
                  << e.what() << std::endl;
      }
    }
  };

  // Initialize device, queue and context
  cl::sycl::device dev;
  // Initialize device, queue and context
  if (!ctx_) {
    dev = fastcalosycl::syclcommon::GetTargetDevice();
    // dev = cl::sycl::device(cl::sycl::default_selector());
    queue_ = cl::sycl::queue(dev);
    ctx_ = new cl::sycl::context(queue_.get_context());
  } else {
    dev = ctx_->get_devices()[0];
    queue_ = cl::sycl::queue(*ctx_, dev);
  }

#ifndef SYCL_TARGET_HIP
  // Ensure device can handle USM device allocations.
  // N.B. Not supported by syclcc (Mar. 22, 2021)
  if (!dev.get_info<cl::sycl::info::device::usm_device_allocations>()) {
    std::cout << "ERROR :: device \""
              << dev.get_info<cl::sycl::info::device::name>()
              << "\" does not support usm_device_allocations!" << std::endl;
    return false;
  }
#endif

  // Name of the device running on
  std::string dev_name = dev.get_info<cl::sycl::info::device::name>();
  std::cout << "SyclGeo::Geo  Using device \"" << dev_name << "\"" << std::endl;

  //
  // CELLS
  //

  if (!AllocMemCells(&dev)) {
    std::cout << "ERROR :: Unable to allocate memory!" << std::endl;
    return false;
  }

  // Assign arrays for the host-side cells.
  int cell_index = 0;
  for (cellmap_t::iterator ic = cell_map_->begin(); ic != cell_map_->end();
       ++ic) {
    cells_[cell_index] = *(*ic).second;
    Identifier id = ((*ic).second)->identify();
    cell_id_[cell_index++] = id;
  }

  // Copy cell data to the device.
  std::cout << "Copying host cells to device... " << std::endl;

  // Start timer.
  auto geo_cpy_start = std::chrono::system_clock::now();
  queue_
      .memcpy(cells_device_, &cells_[0],
              num_cells_ * sizeof(CaloDetDescrElement));
  // End timer.
  auto geo_cpy_end = std::chrono::system_clock::now();
  // Time to copy geometry host->device.
  auto geo_cpy_dur = std::chrono::duration<double>(geo_cpy_end - geo_cpy_start);
  std::cout << "\tCells copied in " << std::dec << geo_cpy_dur.count()
            << " ms.\n"
            << std::endl;

  //
  // REGIONS
  //

  // Allocate device memory for each sampling's regions
  SampleIndex* device_si = (SampleIndex*)malloc_device(
      max_sample_ * sizeof(*sample_index_), dev, *ctx_);

  // Copy sampling array to device
  queue_
      .memcpy(device_si, &sample_index_[0],
              max_sample_ * sizeof(*sample_index_));

  for (unsigned int iregion = 0; iregion < num_regions_; ++iregion) {
    int num_cells_eta = regions_[iregion].cell_grid_eta();
    int num_cells_phi = regions_[iregion].cell_grid_phi();

    // Allocate device memory for region cells
    long long* region_cells = (long long*)malloc_device(
        num_cells_eta * num_cells_phi * sizeof(long long), dev, *ctx_);
    // Copy region cells to device
    queue_
        .memcpy(region_cells, &regions_[iregion].cell_grid()[0],
                num_cells_eta * num_cells_phi * sizeof(long long));

    // Set cells in this region.
    regions_[iregion].set_cell_grid_device(region_cells);
    // Set pointer to all cells before copying to GPU so we have a reference to
    // their address.
    regions_[iregion].set_cells_device(cells_device_);
  }

  // Allocate region data memory and copy to device
  regions_device_ =
      (GeoRegion*)malloc_device(num_regions_ * sizeof(*regions_), dev, *ctx_);
  queue_.memcpy(regions_device_, &regions_[0], num_regions_ * sizeof(*regions_));

  // Device geometry
  DeviceGeo device_geo{nullptr, nullptr, 0, 0, 0, nullptr};
  device_geo.cells = cells_device_;
  device_geo.num_cells = num_cells_;
  device_geo.num_regions = num_regions_;
  device_geo.regions = regions_device_;
  device_geo.sample_max = max_sample_;
  device_geo.sample_index = device_si;

  // Copy device geometry to device, and set static member variable to the
  // corresponding pointer.
  DeviceGeo* device_geo_ptr =
      (DeviceGeo*)malloc_device(sizeof(device_geo), dev, *ctx_);
  queue_.memcpy(device_geo_ptr, &device_geo, sizeof(device_geo)).wait();
  DevGeo = device_geo_ptr;

  return true;
}  // LoadDeviceGeo

const CaloDetDescrElement* Geo::FindCell(Identifier id) {
  auto cell = cell_map_->find(id);
  if (cell == cell_map_->end()) {
    return nullptr;
  } else {
    return cell->second;
  }
}

long long Geo::GetCell(unsigned int sampling, float eta, float phi) {
  float distance = 10000000;
  long long best_cell = -1LL;

  // Check if we have a valid sampling
  if (sampling >= max_sample_) {
    return -1;
  }

  // Get SampleIndex info
  unsigned int sample_size = sample_index_[sampling].size;
  int sample_index = sample_index_[sampling].index;

  if (sample_size < 1) {
    return -1;
  }

  if (sampling < 21) {  // Excluding FCal samplings
    for (int skip_range_check = 0; skip_range_check <= 1; ++skip_range_check) {
      for (unsigned int j = sample_index; j < sample_index + sample_size; ++j) {
        if (!skip_range_check) {
          if (eta < regions_[j].min_eta()) {
            continue;
          }
          if (eta > regions_[j].max_eta()) {
            continue;
          }
        }
        unsigned int int_steps = 0;
        float new_dist = 0;
        long long new_cell =
            regions_[j].get_cell(eta, phi, &new_dist, &int_steps);
        if (new_dist < distance) {
          best_cell = new_cell;
          distance = new_dist;
          if (new_dist < -0.1) {  // We are well within the cell, take it.
            break;
          }
        }
      }
      if (best_cell >= 0) {
        break;
      }
    }
  } else {
    // FCal sampling
    return -2;
  }
  return best_cell;
}

bool Geo::CopyGeoRegion(CaloGeometryLookup* gl, GeoRegion* gr) {
  unsigned int num_eta = gl->cell_grid_eta();
  unsigned int num_phi = gl->cell_grid_phi();
  gr->set_xy_grid_adjust(gl->xy_grid_adjustment_factor());
  gr->set_index(gl->index());
  gr->set_cell_grid_eta(num_eta);
  gr->set_cell_grid_phi(num_phi);
  gr->set_min_eta(gl->mineta());
  gr->set_min_phi(gl->minphi());
  gr->set_max_eta(gl->maxeta());
  gr->set_max_phi(gl->maxphi());
  gr->set_deta(gl->deta());
  gr->set_dphi(gl->dphi());
  gr->set_min_eta_raw(gl->mineta_raw());
  gr->set_min_phi_raw(gl->minphi_raw());
  gr->set_max_eta_raw(gl->maxeta_raw());
  gr->set_max_phi_raw(gl->maxphi_raw());
  gr->set_eta_corr(gl->eta_correction());
  gr->set_phi_corr(gl->phi_correction());
  gr->set_min_eta_corr(gl->mineta_correction());
  gr->set_min_phi_corr(gl->minphi_correction());
  gr->set_max_eta_corr(gl->maxeta_correction());
  gr->set_max_phi_corr(gl->maxphi_correction());
  gr->set_deta_double(gl->deta_double());
  gr->set_dphi_double(gl->dphi_double());

  // Now load cells
  long long* cells = (long long*)malloc(sizeof(long long) * num_eta * num_phi);
  gr->set_cell_grid(cells);

  if (num_eta != (*(gl->cell_grid())).size()) {
    std::cout << "ERROR :: Number of eta regions mismatch!" << std::endl
              << "num_eta: " << num_eta
              << ", expected: " << (*gl->cell_grid()).size() << std::endl;
    return false;
  }

  // Loop over eta
  for (unsigned int ieta = 0; ieta < num_eta; ++ieta) {
    if (num_phi != (*(gl->cell_grid()))[ieta].size()) {
      std::cout << "ERROR :: Number of phi regions mismatch!" << std::endl
                << "num_phi : " << num_phi
                << ", expected: " << (*gl->cell_grid())[ieta].size()
                << std::endl;
      return false;
    }
    // Loop over phi
    for (unsigned int iphi = 0; iphi < num_phi; ++iphi) {
      auto c = (*(gl->cell_grid()))[ieta][iphi];
      if (c) {
        cells[ieta * num_phi + iphi] = c->calo_hash();
      } else {
        cells[ieta * num_phi + iphi] = -1;
      }
    }  // Loop over phi
  }    // Loop over eta
  return true;
}  // copy_geo_region