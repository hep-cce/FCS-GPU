// Copyright (C) 2002-2020 CERN for the benefit of the ATLAS collaboration

// Storage of passive simulation data used during on-device simulation.
// These properties are set by different class objects before being transferred
// to the SYCL device for processing.

#ifndef FASTCALOSYCL_SYCLCOMMON_PROPS_H_
#define FASTCALOSYCL_SYCLCOMMON_PROPS_H_

#include <SyclCommon/Histo.h>

namespace fastcalosycl::syclcommon {

static const unsigned int kMinHits = 256;
static const unsigned int kMaxHits = 200000;
static const unsigned int kMaxBins = 1024;
static const unsigned int kMaxUniqueHits = 2000;

struct CellProps {
  unsigned long cell_id;
  float energy;
};

struct SimProps {
  // Particle properties
  int pdgId;
  double charge;

  // TFCSSimulationState properties
  bool is_first_event;
  bool is_last_event;

  // TFCSExtrapolationState properties
  float extrap_eta_ent;
  float extrap_phi_ent;
  float extrap_r_ent;
  float extrap_z_ent;
  float extrap_eta_exit;
  float extrap_phi_exit;
  float extrap_r_exit;
  float extrap_z_exit;
  float extrap_weight;
  float energy;

  // TFCSHistoLateralShapeParametrization properties
  int calo_sampling;
  bool is_phi_symmetric;

  // TFCSLateralShapeParametrizationHitChain properties
  int num_hits;

  // SimHitRng properties
  unsigned long num_cells;
  float* random_nums;              // Array holding random numbers
  float* cell_energy;              // Array holding energy of all cells
  unsigned int hits_in_event;      // Index/counter for cell_energy array
  CellProps* hit_cell_energy;      // Host-side array for hit cells
  CellProps* hit_cell_energy_dev;  // Device-side array for hit cells
  int* num_unique_hits;            // Number of unique hit cells
  unsigned int max_unique_hits;    // Maximum number of hit cells
  Histo1DFunction* h1df_dev;
  Histo2DFunction* h2df_dev;
  cl::sycl::queue* queue;

  void* rng;    // The RNG
  void* histo;  // Histo{1,2}DFunction
  void* devgeo;
};

}  // namespace fastcalosycl::syclcommon
#endif  // FASTCALOSYCL_SYCLCOMMON_PROPS_H_
