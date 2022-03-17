// Copyright (C) 2002-2020 CERN for the benefit of the ATLAS collaboration

#ifndef FASTCALOSYCL_SYCLGEO_GEO_H_
#define FASTCALOSYCL_SYCLGEO_GEO_H_

#include <SyclCommon/DeviceCommon.h>
#include <SyclGeo/GeoRegion.h>

#include <CL/sycl.hpp>
#include <map>

#include "CaloDetDescrElement.h"
#include "CaloGeometryFromFile.h"

typedef std::map<Identifier, const CaloDetDescrElement*> cellmap_t;

// Stores information about a given sampling; sampling index and the "size",
// i.e. number of regions in the sampling.
struct SampleIndex {
  unsigned int size;
  int index;
};

// Geometry information to reside on device.
struct DeviceGeo {
  CaloDetDescrElement* cells;
  GeoRegion* regions;
  unsigned long num_cells;
  unsigned int num_regions;
  int sample_max;
  SampleIndex* sample_index;
};

// Geometry class.
// Stores calorimeter geometry information -- e.g. regions, cells, etc. -- that
// is read in from a CaloGeometryFromFile object. The data structure is designed
// to be compatible for a SYCL device, and can therefore be loaded to an
// accelerator.
// Host and device memory are allocated, and clients can access both the host
// and device allocated memory through accessor functions.
// The memory is freed upon destruction.
class Geo {
 public:
  Geo();
  Geo(cl::sycl::context* ctx);
  ~Geo();

  // Retrieve the DeviceGeo member variable.
  // All access to the returned data needs to be done within a kernel.
  static struct DeviceGeo* DevGeo;

  void set_num_cells(unsigned long num_cells) { num_cells_ = num_cells; }
  void set_num_regions(unsigned int num_regions) { num_regions_ = num_regions; }
  void set_cell_map(cellmap_t* cell_map) { cell_map_ = cell_map; }
  void set_host_regions(GeoRegion* regions) { regions_ = regions; }
  void set_device_regions(GeoRegion* regions) { regions_device_ = regions; }
  void set_host_cells(CaloDetDescrElement* cells) { cells_ = cells; }
  void set_device_cells(CaloDetDescrElement* cells) { cells_device_ = cells; }
  void set_max_sample(int sample) { max_sample_ = sample; }
  void set_sample_index(SampleIndex* index) { sample_index_ = index; }
  const CaloDetDescrElement* index_to_cell(unsigned long index) {
    return (*cell_map_)[cell_id_[index]];
  }
  unsigned long GetNumCells() const { return num_cells_; }
  unsigned long GetNumRegions() const { return num_regions_; }
  const CaloDetDescrElement* GetCells() const { return cells_; }
  const CaloDetDescrElement* GetCellsDevice() const { return cells_device_; }

  // Initialize geometry.
  // Performs memory allocation for storing geometry on the host device.
  // Takes a CaloGeometryFromFile and assigns the relevant data -- cells,
  // regions and samplings -- to this class member.
  bool Init(CaloGeometryFromFile* cg);

  // Copy geometry to device using USM. Returns true if device memory is
  // allocated and data is copied, and false otherwise.
  bool LoadDeviceGeo();

  // Retrieve the cl::sycl::queue associated to this Geo.
  // All device-side data management is done through the returned object.
  cl::sycl::queue& GetGeoDeviceQueue() { return queue_; }

  // Retrieve a given cell by sampling, eta and phi.
  long long GetCell(unsigned int sampling, float eta, float phi);

  // Finds a cell from cell_map_ given an Identifier.
  // Returns nullptr if the Identifier is invalid, i.e. there is no
  // corresponding cell.
  const CaloDetDescrElement* FindCell(Identifier id);

  // Loads calorimeter geometry.
  // Requires a valid input directory be set that contains a "CaloGeometry"
  // directory:
  // -- ${FCS_INPUT_PATH}/
  //    -- CaloGeometry/
  //        -- FCal{1,2,3}-electrodes.<...>.dat
  //        -- Geometry-ATLAS-<...>.root
  //        -- cellId_vs_cellHashId_map.txt
  // Returns a CaloGeometryFromFile object.
  // CaloGeometryFromFile* GetCaloGeo();

 private:
  bool AllocMemCells(cl::sycl::device* dev);

  // Copies data from a CaloGeometryLookup to a GeoRegion.
  // This reduces the amount of data transferred to the SYCL device.
  bool CopyGeoRegion(CaloGeometryLookup* gl, GeoRegion* gr);

  unsigned long num_cells_;            // Number of cells
  unsigned int num_regions_;           // Number of regions
  cellmap_t* cell_map_;                // From CaloGeometry class
  GeoRegion* regions_;                 // Regions on host
  GeoRegion* regions_device_;          // Regions on device
  CaloDetDescrElement* cells_;         // Cells on host
  CaloDetDescrElement* cells_device_;  // Cells on device
  Identifier* cell_id_;                // Cell ID to Identifier lookup table
  unsigned int max_sample_;            // Max number of samples
  SampleIndex* sample_index_;          // Sample information

  // SYCL-related members
  cl::sycl::queue queue_;   // SYCL queue; needed for context
  cl::sycl::context* ctx_;  // SYCL device context; needed for freeing memory in
                            // the destructor
};
#endif  // FASTCALOSYCL_SYCLGEO_GEO_H_
