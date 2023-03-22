// Copyright (C) 2002-2020 CERN for the benefit of the ATLAS collaboration

#include <FastCaloSycl/SimEvent.h>
#include <SyclCommon/DeviceCommon.h>
#include <SyclGeo/Geo.h>

#define PI 3.14159265358979323846f
#define TWOPI 2 * 3.14159265358979323846f

/*
double Phi_mpi_pi(double x) {
  // TODO: Check for NaN
  while (x >= PI) {
    x -= TWOPI;
  }
  while (x < -PI) {
    x += TWOPI;
  }
  return x;
}

SimEvent::SimEvent(float energy, unsigned int nhits,
                   syclcommon::SimProps* props)
    : energy_(energy),
      num_hits_(nhits),
      simprops_(props),
      is_initialized_(false) {
  // devgeo_ = (DeviceGeo*)simprops_->devgeo;
  // rng_ = (SimHitRng*)simprops_->rng;
  // simprops_->random_nums = rng_->random_nums_ptr(nhits);
  // simprops_->num_unique_hits = rng_->get_num_unique_hits();
}

// void SimEvent::operator()(cl::sycl::item<1> item) {
// Rand4Hits* rd4h = (Rand4Hits*)args.rd4h;

// float* r = rd4h->rand_ptr(nhits);

// rd4h->add_a_hits(nhits);
// args.rand = r;

// unsigned long ncells = args.ncells;
// args.maxhitct = MAXHITCT;

// args.cells_energy = rd4h->get_cells_energy();  // Hit cell energy map ,
// size
//                                                // of ncells(~200k float)
// args.hitcells_E = rd4h->get_cell_e();  // Hit cell energy map, moved
// together args.hitcells_E_h = rd4h->get_cell_e_h();  // Host array

// args.hitcells_ct = rd4h->get_ct();  // single value, number of  uniq hit
// cells

// cudaError_t err = cudaGetLastError();

// int blocksize = BLOCK_SIZE;
// int threads_tot = args.ncells;
// int nblocks = (threads_tot + blocksize - 1) / blocksize;

// simulate_clean<<<nblocks, blocksize>>>(args);

// blocksize = BLOCK_SIZE;
// threads_tot = nhits;
// nblocks = (threads_tot + blocksize - 1) / blocksize;
// simulate_A<<<nblocks, blocksize>>>(E, nhits, args);
// nblocks = (ncells + blocksize - 1) / blocksize;
// simulate_ct<<<nblocks, blocksize>>>(args);

// int ct;
// gpuQ(cudaMemcpy(&ct, args.hitcells_ct, sizeof(int),
// cudaMemcpyDeviceToHost)); gpuQ(cudaMemcpy(args.hitcells_E_h,
// args.hitcells_E, ct * sizeof(Cell_E),
//                 cudaMemcpyDeviceToHost));

// // pass result back
// args.ct = ct;
// }

bool SimEvent::Init() {
  if (is_initialized_) {
    return true;
  }

  // simprops_ = &props;
  // histo_ = (syclcommon::Histo*)props.histo;
  // rng_ = (SimHitRng*)props.rng;
  // float* r = rng_->random_nums(nhits);
  // rng_->add_current_hits(nhits);
  // Rand4Hits* rd4h = (Rand4Hits*)args.rd4h;

  // float* r = rd4h->rand_ptr(nhits);

  // rd4h->add_a_hits(nhits);
  // args.rand = r;

  // unsigned long ncells = args.ncells;
  // args.maxhitct = MAXHITCT;

  // args.cells_energy = rd4h->get_cells_energy();  // Hit cell energy map ,
  // size
  //                                                // of ncells(~200k float)
  // args.hitcells_E = rd4h->get_cell_e();  // Hit cell energy map, moved
  // together args.hitcells_E_h = rd4h->get_cell_e_h();  // Host array

  // args.hitcells_ct = rd4h->get_ct();  // single value, number of  uniq hit
  // cells

  // cudaError_t err = cudaGetLastError();

  // int blocksize = BLOCK_SIZE;
  // int threads_tot = args.ncells;
  // int nblocks = (threads_tot + blocksize - 1) / blocksize;

  // simulate_clean<<<nblocks, blocksize>>>(args);

  // blocksize = BLOCK_SIZE;
  // threads_tot = nhits;
  // nblocks = (threads_tot + blocksize - 1) / blocksize;
  // simulate_A<<<nblocks, blocksize>>>(E, nhits, args);
  // nblocks = (ncells + blocksize - 1) / blocksize;
  // simulate_ct<<<nblocks, blocksize>>>(args);

  // int ct;
  // gpuQ(cudaMemcpy(&ct, args.hitcells_ct, sizeof(int),
  // cudaMemcpyDeviceToHost)); gpuQ(cudaMemcpy(args.hitcells_E_h,
  // args.hitcells_E, ct * sizeof(Cell_E),
  //                 cudaMemcpyDeviceToHost));

  // // pass result back
  // args.ct = ct;

  is_initialized_ = true;
  return is_initialized_;
}

// void SimEvent::SimulateClean(cl::sycl::nd_item<1> item) const {
//   unsigned long wid =
//       item.get_local_id(0) + (item.get_group(0) * item.get_local_range(0));
//   float* cells_energy = rng_->get_cells_energy();
//   if (wid < devgeo_->num_cells) {
//     if (wid % 1000 == 0) {
//       cells_energy[wid] = 0.0;
//       // cl::sycl::intel::experimental::printf(
//       //     fastcalosycl::syclcommon::kTestCellEnergy, wid,
//       cells_energy[wid]);
//     }
//   }
//   if (wid == 0) {
//     simprops_->num_unique_hits[0] = 0;
//   }
// }

// void SimEvent::SimulateShapeAndWiggle(cl::sycl::nd_item<1> item) const {
//   unsigned long wid =
//       item.get_local_id(0) + (item.get_group(0) * item.get_local_range(0));
//   if (wid < num_hits_) {
//     SimHit hit;
//     hit.set_E(energy_);
//     CalcHitCenterPosition(hit);
//     HistoLateralShapeParam(hit, wid);
//     HitCellMappingWiggle(hit, wid);
//   }
// }

void SimEvent::SimulateUnique(cl::sycl::nd_item<1> item) const {
  unsigned long wid =
      item.get_local_id(0) + (item.get_group(0) * item.get_local_range(0));
  if (wid < simprops_->num_cells) {
    if (simprops_->cell_energy[wid] > 0) {
      auto unique_hits =
          sycl::intel::atomic_ref<int, sycl::intel::memory_order::relaxed,
                                  sycl::intel::memory_scope::device,
                                  sycl::access::address_space::global_space>(
              (*simprops_->num_unique_hits));
      int num_unique = unique_hits.fetch_add(1);
      syclcommon::CellProps cell_props = {0, 0};
      cell_props.cell_id = wid;
      cell_props.energy = simprops_->cell_energy[wid];
      simprops_->hit_cell_energy_dev[num_unique] = cell_props;
    }
  }
}

// void SimEvent::CalcHitCenterPosition(SimHit& hit) const {
//   hit.set_center_r((1.0 - simprops_->extrap_weight) * simprops_->extrap_r_ent
//   +
//                    (simprops_->extrap_weight * simprops_->extrap_r_exit));
//   hit.set_center_z((1.0 - simprops_->extrap_weight) * simprops_->extrap_z_ent
//   +
//                    (simprops_->extrap_weight * simprops_->extrap_z_exit));
//   hit.set_center_eta((1.0 - simprops_->extrap_weight) *
//                          simprops_->extrap_eta_ent +
//                      (simprops_->extrap_weight *
//                      simprops_->extrap_eta_exit));
//   hit.set_center_phi((1.0 - simprops_->extrap_weight) *
//                          simprops_->extrap_phi_ent +
//                      (simprops_->extrap_weight *
//                      simprops_->extrap_phi_exit));
// }

// void SimEvent::HistoLateralShapeParam(SimHit& hit, unsigned long wid) const {
//   float charge = simprops_->charge;
//   float center_r = hit.center_r();
//   float center_z = hit.center_z();
//   float center_eta = hit.center_eta();
//   float center_phi = hit.center_phi();

//   float rnd1 = simprops_->random_nums[wid];
//   float rnd2 = simprops_->random_nums[wid + simprops_->num_hits];
//   float alpha = 0.0;
//   float r = 0.0;
//   if (simprops_->is_phi_symmetric) {
//     if (rnd2 >= 0.5) {
//       // Fill negative-phi half
//       rnd2 -= 0.5;
//       rnd2 *= 2.0;
//       RandomToH2DF(alpha, r, rnd1, rnd2);
//       alpha = -alpha;
//     } else {
//       rnd2 *= 2;
//       RandomToH2DF(alpha, r, rnd1, rnd2);
//     }
//   } else {
//     RandomToH2DF(alpha, r, rnd1, rnd2);
//   }

//   float deta_mm = r * cl::sycl::cos(alpha);
//   float dphi_mm = r * cl::sycl::sin(alpha);

//   // Particles with negative eta are expected to have the same shape as those
//   // with positive eta after the transformation: deta -> -deta
//   if (center_eta < 0.0) {
//     deta_mm = -deta_mm;
//   }
//   // Particles with negative eta are expected to have the same shape as those
//   // with positive eta after the transformation: deta -> -deta
//   if (charge < 0.0) {
//     dphi_mm = -dphi_mm;
//   }

//   float dist0 = cl::sycl::sqrt(center_r * center_r + center_z * center_z);
//   float jacobian_eta = cl::sycl::abs(2.0 * cl::sycl::abs(-center_eta) /
//                                      (1.0 + cl::sycl::exp(-2 * center_eta)));
//   float deta = deta_mm / jacobian_eta / dist0;
//   float dphi = dphi_mm / center_r;
//   hit.set_be_hit(center_eta + deta, center_phi + dphi, center_z, hit.E());
// }
*/
