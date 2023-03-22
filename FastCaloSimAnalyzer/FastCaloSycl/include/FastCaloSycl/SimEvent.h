// Copyright (C) 2002-2020 CERN for the benefit of the ATLAS collaboration

#ifndef FASTCALOSYCL_SIMEVENT_H_
#define FASTCALOSYCL_SIMEVENT_H_

#include <FastCaloSycl/SimHit.h>
#include <SyclCommon/Histo.h>
#include <SyclCommon/Props.h>
#include <SyclGeo/Geo.h>
#include <SyclRng/SimHitRng.h>

#define HIPSYCL_EXT_FP_ATOMICS
#include <CL/sycl.hpp>

#define PI 3.14159265358979323846f
#define TWOPI 2 * 3.14159265358979323846f

namespace syclcommon = fastcalosycl::syclcommon;

// Performs simulation.
// Class is a callable that should be instantiated inside a
// cl::sycl::queue::submit(), and then called from a command group handler
// kernel dispatch function, e.g. cl::sycl::handler::parallel_for<...>().
/*
class SimEvent {
 public:
  SimEvent() = delete;
  SimEvent(float energy, unsigned int nhits, syclcommon::SimProps* props);
  // ~SimEvent();

  // Simulation kernel function object.
  // Makes calls to private functions.
  // SYCL_EXTERNAL void operator()(cl::sycl::item<1> item);

  // SYCL_EXTERNAL void SimulateShapeAndWiggle(cl::sycl::nd_item<1> item) const;

 private:
  bool Init();
  SYCL_EXTERNAL void SimulateUnique(cl::sycl::nd_item<1> item) const;
  // SYCL_EXTERNAL void HistoLateralShapeParam(SimHit& hit,
  //                                           unsigned long wid) const;
  SYCL_EXTERNAL void HitCellMappingWiggle(SimHit& hit, unsigned long wid) const;
  SYCL_EXTERNAL void HitCellMapping(SimHit& hit) const;
  SYCL_EXTERNAL long long GetDDE(float eta, float phi) const;

  // Histogram-specific functions
  SYCL_EXTERNAL float RandomToH1DF(float rand, unsigned int bin) const;
  SYCL_EXTERNAL unsigned int FindIndexH1DF(uint32_t rand,
                                           unsigned int bin) const;

  float energy_;
  unsigned int num_hits_;
  bool is_initialized_;
  syclcommon::SimProps* simprops_;
  const DeviceGeo* devgeo_;
  SimHitRng* rng_;
  syclcommon::Histo* histo_;
};
*/

// struct Kernel_SimClean {
//   unsigned long num_cells;
//   int* num_unique_hits;
//   float* cell_energy;

//   void operator()(cl::sycl::nd_item<1> item) {
//     unsigned long wid =
//         item.get_local_id(0) + (item.get_group(0) * item.get_local_range(0));
//     if (wid < num_cells) {
//       cell_energy[wid] = 0.0;
//     }
//     if (wid == 0) {
//       num_unique_hits = 0;
//     }
//   }
// };

/*
struct Kernel_SimClean {
  syclcommon::SimProps* props;

  void operator()(cl::sycl::nd_item<1> item) {
    unsigned long wid =
        item.get_local_id(0) + (item.get_group(0) * item.get_local_range(0));
    // if (wid < props.num_cells) {
    //   props.cell_energy[wid] = 0.0;
    // }
    // if (wid == 0) {
    //   props.num_unique_hits = 0;
    // }
  }
};
*/

class SimResetKernel {
 public:
  SimResetKernel() = delete;
  SimResetKernel(syclcommon::SimProps* props)
      : num_cells_(props->num_cells), num_unique_hits_(nullptr) {
    SimHitRng* rng = (SimHitRng*)props->rng;
    num_unique_hits_ = rng->get_num_unique_hits();
    cells_energy_ = rng->get_cells_energy();
  }

  void operator()(cl::sycl::id<1> id) const {
    unsigned int wid = id.get(0);
    if (wid < num_cells_) {
      cells_energy_[wid] = 0.0;
    }
    if (wid == 0) {
      *num_unique_hits_ = 0;
    }
  }

 private:
  const unsigned long num_cells_;
  int* num_unique_hits_;
  float* cells_energy_;
};

class SimShapeKernel {
 public:
  SimShapeKernel() = delete;
  SimShapeKernel(syclcommon::SimProps* props, float energy)
      : num_hits_(props->num_hits),
        hit_energy_(energy),
        extrap_weight_(props->extrap_weight),
        extrap_r_ent_(props->extrap_r_ent),
        extrap_r_exit_(props->extrap_r_exit),
        extrap_z_ent_(props->extrap_z_ent),
        extrap_z_exit_(props->extrap_z_exit),
        extrap_eta_ent_(props->extrap_eta_ent),
        extrap_eta_exit_(props->extrap_eta_exit),
        extrap_phi_ent_(props->extrap_phi_ent),
        extrap_phi_exit_(props->extrap_phi_exit),
        charge_(props->charge),
        random_nums_(nullptr),
        is_phi_symmetric_(props->is_phi_symmetric),
        h1df_(props->h1df_dev),
        h2df_(props->h2df_dev),
        geo_((DeviceGeo*)props->devgeo),
        calo_sampling_(props->calo_sampling) {
    SimHitRng* rng = (SimHitRng*)props->rng;
    float* rn = rng->random_nums_ptr(num_hits_);
    rng->add_current_hits(num_hits_);
    cells_energy_ = rng->get_cells_energy();
    random_nums_ = rn;
  }

  void operator()(cl::sycl::id<1> id) const {
    unsigned int wid = id.get(0);

    if (wid < num_hits_) {
      SimHit hit;
      hit.set_E(hit_energy_);
      CalcHitCenterPosition(hit);
      HistoLateralShapeParam(hit, wid);
      // if (cl::sycl::abs(hit.eta()) > 10.0 || cl::sycl::abs(hit.phi()) > 10.0)
      // {
      //   return;
      // }
      HitCellMappingWiggle(hit, wid);
    }
  }

 private:
  void CalcHitCenterPosition(SimHit& hit) const {
    hit.set_center_r((1.0f - extrap_weight_) * extrap_r_ent_ +
                     (extrap_weight_ * extrap_r_exit_));
    hit.set_center_z((1.0f - extrap_weight_) * extrap_z_ent_ +
                     (extrap_weight_ * extrap_z_exit_));
    hit.set_center_eta((1.0f - extrap_weight_) * extrap_eta_ent_ +
                       (extrap_weight_ * extrap_eta_exit_));
    hit.set_center_phi((1.0f - extrap_weight_) * extrap_phi_ent_ +
                       (extrap_weight_ * extrap_phi_exit_));
  }

  void HistoLateralShapeParam(SimHit& hit,
                                            unsigned long wid) const {
    float charge = charge_;
    float center_r = hit.center_r();
    float center_z = hit.center_z();
    float center_eta = hit.center_eta();
    float center_phi = hit.center_phi();

    float rnd1 = random_nums_[wid];
    float rnd2 = random_nums_[wid];
    float alpha = 0.0;
    float r = 0.0;
    if (is_phi_symmetric_) {
      if (rnd2 >= 0.5f) {
        // Fill negative-phi half
        rnd2 -= 0.5;
        rnd2 *= 2.0;
        RandomToH2DF(alpha, r, rnd1, rnd2);
        alpha = -alpha;
      } else {
        rnd2 *= 2;
        RandomToH2DF(alpha, r, rnd1, rnd2);
      }
    } else {
      RandomToH2DF(alpha, r, rnd1, rnd2);
    }

    float deta_mm = r * cl::sycl::cos(alpha);
    float dphi_mm = r * cl::sycl::sin(alpha);

    // Particles with negative eta are expected to have the same shape as
    // those with positive eta after the transformation: deta -> -deta
    if (center_eta < 0.0f) {
      deta_mm = -deta_mm;
    }
    // Particles with negative eta are expected to have the same shape as
    // those with positive eta after the transformation: dphi -> -dphi
    if (charge < 0.0f) {
      dphi_mm = -dphi_mm;
    }

    float dist0 = cl::sycl::sqrt(center_r * center_r + center_z * center_z);
    float jacobian_eta = cl::sycl::fabs(2.0f * cl::sycl::fabs(-center_eta) /
                                       (1.0f + cl::sycl::exp(-2 * center_eta)));
    float deta = deta_mm / jacobian_eta / dist0;
    float dphi = dphi_mm / center_r;
    hit.set_be_hit(center_eta + deta, center_phi + dphi, center_z, hit.E());
    // if (hit.phi() > 5.0) {
    //   cl::sycl::intel::experimental::printf(
    //       fastcalosycl::syclcommon::kPrintIdEtaPhi, wid, hit.eta(),
    //       hit.phi());
    // }
  }

  void RandomToH2DF(float& val_x, float& val_y, float rand_x,
                                  float rand_y) const {
    int ibin = FindIndexH2DF(h2df_->num_binsx * h2df_->num_binsy, rand_x);
    int biny = ibin / h2df_->num_binsx;
    int binx = ibin - (h2df_->num_binsx * biny);

    float base_content = 0.0;
    if (ibin > 0) {
      base_content = h2df_->contents[ibin - 1];
    }

    float d_content = h2df_->contents[ibin] - base_content;
    if (d_content > 0) {
      val_x = h2df_->bordersx[binx] +
              (h2df_->bordersx[binx + 1] - h2df_->bordersx[binx]) *
                  (rand_x - base_content) / d_content;
    } else {
      val_x = h2df_->bordersx[binx] +
              (h2df_->bordersx[binx + 1] - h2df_->bordersx[binx]) / 2;
    }
    val_y = h2df_->bordersy[biny] +
            (h2df_->bordersy[biny + 1] - h2df_->bordersy[biny]) * rand_y;
  }

  unsigned int FindIndexH2DF(unsigned int size,
                                           float rand_x) const {
    unsigned int lo = 0;
    unsigned int hi = size - 1;
    unsigned int index = (hi - lo) / 2;
    while (hi != lo) {
      if (rand_x >= h2df_->contents[index]) {
        lo = index + 1;
      } else {
        hi = index;
      }
      index = (hi + lo) / 2;
    }
    return index;
  }

  void HitCellMappingWiggle(SimHit& hit,
                                          unsigned long wid) const {
    unsigned int num_funcs = h1df_->num_funcs;
    float eta = cl::sycl::fabs(hit.eta());
    if (eta < h1df_->low_edge[0] || eta > h1df_->low_edge[num_funcs]) {
      HitCellMapping(wid, hit);
    }
    unsigned int bin = num_funcs;
    for (unsigned int i = 0; i < num_funcs + 1; ++i) {
      if (h1df_->low_edge[i] > eta) {
        bin = i;
        break;
      }
    }
    bin -= 1;
    float rand = random_nums_[wid + (2 * 1)];
    float wiggle = RandomToH1DF(rand, bin);
    float phi_shift = hit.phi() + wiggle;
    hit.set_phi(phi_mpi_pi(phi_shift));
    HitCellMapping(wid, hit);
  }

  void HitCellMapping(unsigned long wid, SimHit& hit) const {
    float eta = hit.eta();
    float phi = hit.phi();
    //! TODO
    long long dde = GetDDE(hit.eta(), hit.phi());
    // long long dde = wid;
    if (dde < 0) {
// #if not defined SYCL_TARGET_CUDA and not defined SYCL_TARGET_HIP
      // cl::sycl::ONEAPI::experimental::printf(
      //     fastcalosycl::syclcommon::kString,
      //     "SimShapeKernel::HitCellMapping()    Cell DDE not found (DDE < 0) !");
      // cl::sycl::ONEAPI::experimental::printf(
      //     fastcalosycl::syclcommon::kPrintIdEtaPhi, wid, hit.eta(), hit.phi());
// #endif
      return;
    }
#ifdef SYCL_DEVICE_ONLY
    cl::sycl::atomic<float> cells_energy{cl::sycl::multi_ptr<float, cl::sycl::access::address_space::global_space>{&cells_energy_[dde]}};
    cells_energy.fetch_add(hit.E());
#endif
  }

  long long GetDDE(float eta, float phi) const {
    if (calo_sampling_ < 0 || calo_sampling_ >= geo_->sample_max) {
// #if not defined SYCL_TARGET_CUDA and not defined SYCL_TARGET_HIP
      // cl::sycl::ONEAPI::experimental::printf(
      //     fastcalosycl::syclcommon::kString,
      //     "SimShapeKernel::GetDDe()    calo_sampling_ out of range!");
// #endif
      return -1L;
    }
    SampleIndex* sindex_ = geo_->sample_index;
    unsigned int sindex_size = sindex_->size;
    int isindex = sindex_->index;
    if (sindex_size == 0) {
// #if not defined SYCL_TARGET_CUDA and not defined SYCL_TARGET_HIP
      // cl::sycl::ONEAPI::experimental::printf(
      //     fastcalosycl::syclcommon::kString,
      //     "SimShapeKernel::GetDDe()    isindex == 0!");
// #endif
      return -2L;
    }
    float dist = 0.0;
    long long best_dde = -100L;
    float* distance = &dist;
    (*distance) = +10000000;
    unsigned int int_steps = 0;
    int best_steps = 0;
    if (calo_sampling_ < 21) {
      for (unsigned int skip_range_check = 0; skip_range_check <= 1;
           ++skip_range_check) {
        for (unsigned int j = isindex; j < isindex + sindex_size; ++j) {
          if (!skip_range_check) {
            if (eta < geo_->regions[j].min_eta()) {
              continue;
            }
            if (eta > geo_->regions[j].max_eta()) {
              continue;
            }
          }
          int_steps = 0;
          float new_dist = 0.0;
          long long new_dde =
              geo_->regions[j].get_cell(eta, phi, &new_dist, &int_steps);
          if (new_dist < (*distance)) {
            best_dde = new_dde;
            (*distance) = new_dist;
            best_steps = int_steps;
            if (new_dist < -0.1f) {
              break;
            }
          }
        }
        if (best_dde >= 0) {
          break;
        }
      }
    } else {
// #if not defined SYCL_TARGET_CUDA and not defined SYCL_TARGET_HIP
      // cl::sycl::ONEAPI::experimental::printf(
      //     fastcalosycl::syclcommon::kString,
      //     "SimShapeKernel::GetDDE()    calo_sampling_ >= 21!");
// #endif
      return -3L;
    }
    return best_dde;
  }

  float RandomToH1DF(float rand, unsigned int bin) const {
    uint32_t uint_rand = h1df_->max_value * rand;
    unsigned int ibin = FindIndexH1DF(uint_rand, bin);
    unsigned int binx = ibin;
    uint32_t base_content = 0;
    if (ibin > 0) {
      base_content = h1df_->contents[bin][ibin - 1];
    }
    uint32_t d_content = h1df_->contents[bin][ibin] - base_content;
    if (d_content > 0) {
      return h1df_->borders[bin][binx] +
             ((h1df_->borders[bin][binx + 1] - h1df_->borders[bin][binx]) *
              (uint_rand - base_content)) /
                 d_content;
    } else {
      return h1df_->borders[bin][binx] +
             (h1df_->borders[bin][binx + 1] - h1df_->borders[bin][binx]) / 2;
    }
  }

  unsigned int FindIndexH1DF(uint32_t rand,
                                           unsigned int bin) const {
    unsigned int lo = 0;
    unsigned int hi = h1df_->num_funcs - 1;
    unsigned int index = (hi - lo) / 2;
    while (hi != lo) {
      if (rand >= h1df_->contents[bin][index]) {
        lo = index + 1;
      } else {
        hi = index;
      }
      index = (lo + hi) / 2;
    }
    return index;
  }

  inline float phi_mpi_pi(float x) const {
    if (x > PI)
      return x - TWOPI;
    else if (x < -PI)
      return x + TWOPI;
    else
      return x;
  }

  const int num_hits_;
  float* cells_energy_;
  const float hit_energy_;
  const float extrap_weight_;
  const float extrap_r_ent_;
  const float extrap_r_exit_;
  const float extrap_z_ent_;
  const float extrap_z_exit_;
  const float extrap_eta_ent_;
  const float extrap_eta_exit_;
  const float extrap_phi_ent_;
  const float extrap_phi_exit_;
  const float charge_;
  const float* random_nums_;
  const bool is_phi_symmetric_;
  const fastcalosycl::syclcommon::Histo1DFunction* h1df_;
  const fastcalosycl::syclcommon::Histo2DFunction* h2df_;
  int calo_sampling_;
  const DeviceGeo* geo_;
};

#endif  // FASTCALOSYCL_SIMEVENT_H_
