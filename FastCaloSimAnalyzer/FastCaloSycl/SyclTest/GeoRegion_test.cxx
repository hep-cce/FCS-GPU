// Copyright (C) 2002-2020 CERN for the benefit of the ATLAS collaboration

#ifdef NDEBUG
#undef NDEBUG
#endif

#include <SyclGeo/Geo.h>
#include <assert.h>

#include <iostream>

#include "CaloGeometryFromFile.h"
#include "TFCSSampleDiscovery.h"

CaloGeometryFromFile* get_calo_geo() {
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

int main() {
  std::cout << "*** GeoRegion_test BEGINS ***" << std::endl;

  // Load the geometry
  CaloGeometryFromFile* calo_geo = get_calo_geo();

  // Initialize geometry regions
  GeoRegion* gr = nullptr;
  unsigned int num_regions = calo_geo->get_tot_regions();
  int* geo_samples = (int*)malloc(CaloGeometry::MAX_SAMPLING * sizeof(int));
  gr = (GeoRegion*)malloc(num_regions * sizeof(GeoRegion));
  assert(gr);

  // Loop over samples
  unsigned int i = 0;
  for (int isamp = 0; isamp < CaloGeometry::MAX_SAMPLING; ++isamp) {
    geo_samples[isamp] = i;
    // Get the number of regions in this sampling
    unsigned int num_regions = calo_geo->get_n_regions(isamp);
    // Loop over regions
    for (unsigned int ireg = 0; ireg < num_regions; ++ireg) {
      CaloGeometryLookup* cgl = calo_geo->get_region(isamp, ireg);
      unsigned int num_eta = cgl->cell_grid_eta();
      unsigned int num_phi = cgl->cell_grid_phi();
      gr[i].set_xy_grid_adjust(cgl->xy_grid_adjustment_factor());
      gr[i].set_index(cgl->index());
      gr[i].set_cell_grid_eta(num_eta);
      gr[i].set_cell_grid_phi(num_phi);
      gr[i].set_min_eta(cgl->mineta());
      gr[i].set_min_phi(cgl->minphi());
      gr[i].set_max_eta(cgl->maxeta());
      gr[i].set_max_phi(cgl->maxphi());
      gr[i].set_deta(cgl->deta());
      gr[i].set_dphi(cgl->dphi());
      gr[i].set_min_eta_raw(cgl->mineta_raw());
      gr[i].set_min_phi_raw(cgl->minphi_raw());
      gr[i].set_max_eta_raw(cgl->maxeta_raw());
      gr[i].set_max_phi_raw(cgl->maxphi_raw());
      gr[i].set_eta_corr(cgl->eta_correction());
      gr[i].set_phi_corr(cgl->phi_correction());
      gr[i].set_min_eta_corr(cgl->mineta_correction());
      gr[i].set_min_phi_corr(cgl->minphi_correction());
      gr[i].set_max_eta_corr(cgl->maxeta_correction());
      gr[i].set_max_phi_corr(cgl->maxphi_correction());
      gr[i].set_deta_double(cgl->deta_double());
      gr[i].set_dphi_double(cgl->dphi_double());

      // Now load cells
      long long* cells =
          (long long*)malloc(sizeof(long long) * num_eta * num_phi);
      gr[i].set_cell_grid(cells);

      if (num_eta != (*(cgl->cell_grid())).size()) {
        std::cout << "num_eta: " << num_eta
                  << ", cell_grid: " << (*cgl->cell_grid()).size() << std::endl;
      }

      // Loop over eta
      unsigned int ncells = 0;
      unsigned int empty_cells = 0;
      for (unsigned int ieta = 0; ieta < num_eta; ++ieta) {
        if (num_phi != (*(cgl->cell_grid()))[ieta].size()) {
          std::cout << "num_phi: " << num_phi
                    << ", cell_grid: " << (*cgl->cell_grid())[ieta].size()
                    << std::endl;
        }
        // Loop over phi
        for (unsigned int iphi = 0; iphi < num_phi; ++iphi) {
          auto c = (*(cgl->cell_grid()))[ieta][iphi];
          if (c) {
            cells[ieta * num_phi + iphi] = c->calo_hash();
            ++ncells;
          } else {
            cells[ieta * num_phi + iphi] = -1;
            ++empty_cells;
          }
        }  // Loop over phi
      }    // Loop over eta
      std::cout << "ncells: " << ncells << ", empty_cells: " << empty_cells
                << std::endl;
      ++i;
    }  // Loop over regions
  }    // Loop over samples

  // Free memory
  delete (gr);
  gr = nullptr;
  delete (calo_geo);
  calo_geo = nullptr;

  std::cout << "*** GeoRegion_test ENDS ***" << std::endl;
  return 0;
}