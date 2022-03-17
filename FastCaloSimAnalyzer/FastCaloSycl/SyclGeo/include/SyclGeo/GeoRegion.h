// Copyright (C) 2002-2020 CERN for the benefit of the ATLAS collaboration

#ifndef FASTCALOSYCL_SYCLGEO_GEOREGION_H_
#define FASTCALOSYCL_SYCLGEO_GEOREGION_H_

#include <CL/sycl.hpp>
#include <memory>

#include "CaloDetDescrElement.h"

#define PI 3.14159265358979323846
#define TWOPI 2 * 3.14159265358979323846

// GeoRegion class
// Cells with given eta and phi make up regions within the calorimeter.
// This class stores such information, and provides functionality to query
// cells.
class GeoRegion {
 public:
  // Constructors
  GeoRegion();

  void set_cells(CaloDetDescrElement* dde);
  void set_cells_device(CaloDetDescrElement* dde);
  void set_cell_grid(long long* cells);
  void set_cell_grid_device(long long* cells);
  void set_index(int i);
  void set_cell_grid_eta(int eta);
  void set_cell_grid_phi(int phi);
  void set_xy_grid_adjust(float adjust);
  void set_deta(float deta);
  void set_dphi(float dphi);
  void set_min_eta(float min_eta);
  void set_min_phi(float min_phi);
  void set_max_eta(float max_eta);
  void set_max_phi(float max_phi);
  void set_min_eta_raw(float min_eta);
  void set_min_phi_raw(float min_phi);
  void set_max_eta_raw(float max_eta);
  void set_max_phi_raw(float max_phi);
  void set_eta_corr(float eta_corr);
  void set_phi_corr(float phi_corr);
  void set_min_eta_corr(float eta_corr);
  void set_min_phi_corr(float phi_corr);
  void set_max_eta_corr(float eta_corr);
  void set_max_phi_corr(float phi_corr);
  void set_deta_double(float deta);
  void set_dphi_double(float dphi);

  long long* cell_grid();
  long long* cell_grid_device();
  CaloDetDescrElement* cells();
  int index();
  SYCL_EXTERNAL int cell_grid_eta();
  SYCL_EXTERNAL int cell_grid_phi();
  int raw_eta_pos_to_index(float eta_raw) const;
  int raw_phi_pos_to_index(float phi_raw) const;
  float min_eta() const;
  SYCL_EXTERNAL float min_phi() const;
  SYCL_EXTERNAL float max_eta() const;
  SYCL_EXTERNAL float max_phi() const;
  SYCL_EXTERNAL float deta() const;
  SYCL_EXTERNAL float dphi() const;
  SYCL_EXTERNAL bool index_range_adjust(int& ieta, int& iphi) const;
  SYCL_EXTERNAL float calc_distance_eta_phi(const long long dde, float eta,
                                            float phi, float& dist_eta0,
                                            float& dist_phi0) const;
  SYCL_EXTERNAL long long get_cell(float eta, float phi, float* distance,
                                   unsigned int* steps) const;

 private:
  SYCL_EXTERNAL float phi_mpi_pi(float x) const;

  long long* cell_grid_;         // Array for calorimeter cells
  long long* cell_grid_device_;  // Array for calorimeter cells on device
  CaloDetDescrElement* cells_;   // Array for detector elements; not owner
  CaloDetDescrElement*
      cells_device_;  // Array for detector elements on device; not owner
  int index_;
  int cell_grid_eta_;
  int cell_grid_phi_;
  float xy_grid_adjust_;
  float deta_;
  float dphi_;
  float min_eta_;
  float min_phi_;
  float max_eta_;
  float max_phi_;
  float min_eta_raw_;
  float min_phi_raw_;
  float max_eta_raw_;
  float max_phi_raw_;
  float eta_corr_;
  float phi_corr_;
  float min_eta_corr_;
  float max_eta_corr_;
  float min_phi_corr_;
  float max_phi_corr_;
  float deta_double_;  // What is "double"?
  float dphi_double_;  // ...
};

#include "SyclGeo/GeoRegion.icc"

#endif  // FASTCALOSYCL_SYCLGEO_GEOREGION_H_
