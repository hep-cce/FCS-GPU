/*
  Copyright (C) 2002-2021 CERN for the benefit of the ATLAS collaboration
*/

#include "CaloGpuGeneral_omp.h"
#include "GeoGpu_structs.h"
#include "GeoRegion.h"
#include "Hit.h"
#include "Rand4Hits.h"

// #include "gpuQ.h"
#include "Args.h"
#include <chrono>
#include <iostream>
#include <omp.h>

#ifdef USE_KOKKOS
#include <Kokkos_Core.hpp>
#include <Kokkos_Random.hpp>
#endif

#define BLOCK_SIZE 256
#define GRID_SIZE 734
#define GRID_SIZE1 5

#define M_PI 3.14159265358979323846
#define M_2PI 6.28318530717958647692

// using namespace CaloGpuGeneral_fnc;

static CaloGpuGeneral::KernelTime timing;

namespace CaloGpuGeneral_omp {

inline long long getDDE(GeoGpu *geo, int sampling, float eta, float phi) {

  float *distance = 0;
  int *steps = 0;

  int MAX_SAMPLING = geo->max_sample;
  Rg_Sample_Index *SampleIdx = geo->sample_index;
  GeoRegion *regions_g = geo->regions;

  if (sampling < 0)
    return -1;
  if (sampling >= MAX_SAMPLING)
    return -1;

  int sample_size = SampleIdx[sampling].size;
  unsigned int sample_index = SampleIdx[sampling].index;

  GeoRegion *gr = (GeoRegion *)regions_g;
  if (sample_size == 0)
    return -1;
  float dist;
  long long bestDDE = -1;
  if (!distance)
    distance = &dist;
  *distance = +10000000;
  int intsteps;
  int beststeps;
  if (steps)
    beststeps = (*steps);
  else
    beststeps = 0;

  if (sampling < 21) {
    for (int skip_range_check = 0; skip_range_check <= 1; ++skip_range_check) {
      for (unsigned int j = sample_index; j < sample_index + sample_size; ++j) {
        if (!skip_range_check) {
          if (eta < gr[j].mineta())
            continue;
          if (eta > gr[j].maxeta())
            continue;
        }
        if (steps)
          intsteps = (*steps);
        else
          intsteps = 0;
        float newdist;
        long long newDDE = gr[j].getDDE(eta, phi, &newdist, &intsteps);
        if (newdist < *distance) {
          bestDDE = newDDE;
          *distance = newdist;
          if (steps)
            beststeps = intsteps;
          if (newdist < -0.1)
            break; // stop, we are well within the hit cell
        }
      }
      if (bestDDE >= 0)
        break;
    }
  } else {
    return -3;
  }
  if (steps)
    *steps = beststeps;

  return bestDDE;
}

/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

inline int find_index_f(float *array, int size, float value) {
  // fist index (from 0)  have element value > value
  // array[i] > value ; array[i-1] <= value
  // std::upbund( )

  int low = 0;
  int high = size - 1;
  int m_index = (high - low) / 2;
  while (high != low) {
    if (value >= array[m_index])
      low = m_index + 1;
    else
      high = m_index;
    m_index = (high + low) / 2;
  }
  return m_index;
}

/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

inline int find_index_uint32(uint32_t *array, int size, uint32_t value) {
  // find the first index of element which has vaule > value
  int low = 0;
  int high = size - 1;
  int m_index = (high - low) / 2;
  while (high != low) {
    if (value > array[m_index])
      low = m_index + 1;
    else if (value == array[m_index]) {
      return m_index + 1;
    } else
      high = m_index;
    m_index = (high - low) / 2 + low;
  }
  return m_index;
}

/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

inline void rnd_to_fct2d(float &valuex, float &valuey, float rnd0, float rnd1,
                         FH2D *hf2d, unsigned long /*i*/, float /*ene*/) {

  int nbinsx = (*hf2d).nbinsx;
  int nbinsy = (*hf2d).nbinsy;
  float *HistoContents = (*hf2d).h_contents;
  float *HistoBorders = (*hf2d).h_bordersx;
  float *HistoBordersy = (*hf2d).h_bordersy;

  /*
   int ibin = nbinsx*nbinsy-1 ;
   for ( int i=0 ; i < nbinsx*nbinsy ; ++i) {
      if   (HistoContents[i]> rnd0 ) {
           ibin = i ;
           break ;
          }
   }
  */
  int ibin = find_index_f(HistoContents, nbinsx * nbinsy, rnd0);

  int biny = ibin / nbinsx;
  int binx = ibin - nbinsx * biny;

  float basecont = 0;
  if (ibin > 0)
    basecont = HistoContents[ibin - 1];

  float dcont = HistoContents[ibin] - basecont;
  if (dcont > 0) {
    valuex =
        HistoBorders[binx] + (HistoBorders[binx + 1] - HistoBorders[binx]) *
                                 (rnd0 - basecont) / dcont;
  } else {
    valuex =
        HistoBorders[binx] + (HistoBorders[binx + 1] - HistoBorders[binx]) / 2;
  }
  valuey = HistoBordersy[biny] +
           (HistoBordersy[biny + 1] - HistoBordersy[biny]) * rnd1;
}

/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

inline float rnd_to_fct1d(float rnd, uint32_t *contents, float *borders,
                          int nbins, uint32_t s_MaxValue) {

  uint32_t int_rnd = s_MaxValue * rnd;
  /*
    int  ibin=nbins-1 ;
    for ( int i=0 ; i < nbins ; ++i) {
      if   (contents[i]> int_rnd ) {
           ibin = i ;
           break ;
          }
    }
  */
  int ibin = find_index_uint32(contents, nbins, int_rnd);

  int binx = ibin;

  uint32_t basecont = 0;
  if (ibin > 0)
    basecont = contents[ibin - 1];

  uint32_t dcont = contents[ibin] - basecont;
  if (dcont > 0) {
    return borders[binx] +
           ((borders[binx + 1] - borders[binx]) * (int_rnd - basecont)) / dcont;
  } else {
    return borders[binx] + (borders[binx + 1] - borders[binx]) / 2;
  }
}

/* * * * * * * * * * * * * * * * * * * * * * * * * */

inline void Rand4Hits_finish(void *rd4h) {

  size_t free, total;
  // gpuQ( cudaMemGetInfo( &free, &total ) );
  std::cout << "TODO GPU memory used(MB): " << (total - free) / 1000000
            << std::endl;
  //    if ( (Rand4Hits*)rd4h ) delete (Rand4Hits*)rd4h;

  std::cout << timing;
}

inline void simulate_A(float E, int nhits, Chain0_Args args) {

  const unsigned long ncells = args.ncells;
  const unsigned long maxhitct = args.maxhitct;

  auto cells_energy = args.cells_energy;
  auto hitcells_ct = args.hitcells_ct;
  auto rand = args.rand;
  auto geo = args.geo;

  int m_initial_device = omp_get_initial_device();

  /************* A **********/

  long t;
// Hit hit;
#pragma omp target is_device_ptr(cells_energy, rand, geo) // map( to : args )
  // #pragma omp target is_device_ptr( cells_energy, rand, geo ) nowait
  // //depend( in : args )
  {
#pragma omp teams distribute                                                   \
    parallel for // num_teams(60) //num_threads(64) //num_teams default 33
    for (t = 0; t < nhits; t++) {
      Hit hit;
      hit.E() = E;

      // CenterPositionCalculation_d( hit, args );
      hit.setCenter_r((1. - args.extrapWeight) * args.extrapol_r_ent +
                      args.extrapWeight * args.extrapol_r_ext);
      hit.setCenter_z((1. - args.extrapWeight) * args.extrapol_z_ent +
                      args.extrapWeight * args.extrapol_z_ext);
      hit.setCenter_eta((1. - args.extrapWeight) * args.extrapol_eta_ent +
                        args.extrapWeight * args.extrapol_eta_ext);
      hit.setCenter_phi((1. - args.extrapWeight) * args.extrapol_phi_ent +
                        args.extrapWeight * args.extrapol_phi_ext);

      // HistoLateralShapeParametrization_d( hit, t, args );
      //  int     pdgId    = args.pdgId;
      float charge = args.charge;

      // int cs=args.charge;
      float center_eta = hit.center_eta();
      float center_phi = hit.center_phi();
      float center_r = hit.center_r();
      float center_z = hit.center_z();

      float alpha, r, rnd1, rnd2;
      rnd1 = rand[t];
      rnd2 = rand[t + args.nhits];
      // printf ( " rands are %f %f ----> \n ", rnd1, rnd2);
      if (args.is_phi_symmetric) {
        if (rnd2 >= 0.5) { // Fill negative phi half of shape
          rnd2 -= 0.5;
          rnd2 *= 2;
          rnd_to_fct2d(alpha, r, rnd1, rnd2, args.fh2d, t, E);
          alpha = -alpha;
        } else { // Fill positive phi half of shape
          rnd2 *= 2;
          rnd_to_fct2d(alpha, r, rnd1, rnd2, args.fh2d, t, E);
        }
      } else {
        rnd_to_fct2d(alpha, r, rnd1, rnd2, args.fh2d, t, E);
      }

      float delta_eta_mm = r * cos(alpha);
      float delta_phi_mm = r * sin(alpha);

      // Particles with negative eta are expected to have the same shape as
      // those with positive eta after transformation: delta_eta --> -delta_eta
      if (center_eta < 0.)
        delta_eta_mm = -delta_eta_mm;
      // Particle with negative charge are expected to have the same shape as
      // positively charged particles after transformation: delta_phi -->
      // -delta_phi
      if (charge < 0.)
        delta_phi_mm = -delta_phi_mm;

      // TODO : save exp and divisions
      float dist000 = sqrt(center_r * center_r + center_z * center_z);
      float eta_jakobi =
          abs(2.0 * exp(-center_eta) / (1.0 + exp(-2 * center_eta)));

      float delta_eta = delta_eta_mm / eta_jakobi / dist000;
      float delta_phi = delta_phi_mm / center_r;

      hit.setEtaPhiZE(center_eta + delta_eta, center_phi + delta_phi, center_z,
                      hit.E());

      // HitCellMappingWiggle_d( hit, args, t, cells_energy );
      int nhist = (*(args.fhs)).nhist;
      float *bin_low_edge = (*(args.fhs)).low_edge;

      float eta = fabs(hit.eta());
      if (eta < bin_low_edge[0] || eta > bin_low_edge[nhist]) {
        // HitCellMapping_d( hit, t, args, cells_energy );
        long long cellele = getDDE(args.geo, args.cs, hit.eta(), hit.phi());
#pragma omp atomic update
        cells_energy[cellele] += (float)(E);
      }

      int bin = nhist;
      for (int i = 0; i < nhist + 1; ++i) {
        if (bin_low_edge[i] > eta) {
          bin = i;
          break;
        }
      }

      //  bin=find_index_f(bin_low_edge, nhist+1, eta ) ;

      bin -= 1;

      unsigned int mxsz = args.fhs->mxsz;
      uint32_t *contents = &(args.fhs->d_contents1D[bin * mxsz]);
      float *borders = &(args.fhs->d_borders1D[bin * mxsz]);
      int h_size = (*(args.fhs)).h_szs[bin];
      uint32_t s_MaxValue = (*(args.fhs)).s_MaxValue;

      float rnd = rand[t + 2 * args.nhits];

      float wiggle = rnd_to_fct1d(rnd, contents, borders, h_size, s_MaxValue);

      float hit_phi_shifted = hit.phi() + wiggle;
      hit.phi() = Phi_mpi_pi(hit_phi_shifted);

      // HitCellMapping
      long long cellele = getDDE(args.geo, args.cs, hit.eta(), hit.phi());
      // printf("t = %ld cellee %lld hit.eta %f hit.phi %f \n", t, cellele,
      // hit.eta(), hit.phi());

#pragma omp atomic update
      //*( cells_energy + cellele ) += E;
      cells_energy[cellele] += (float)(E); // typecast is necessary
    }
  }
  // #pragma omp taskwait
}

inline void simulate_ct(Chain0_Args args) {

  const unsigned long ncells = args.ncells;

  auto cells_energy = args.cells_energy;
  auto hitcells_ct = args.hitcells_ct;
  auto hitcells_E = args.hitcells_E;

#pragma omp target is_device_ptr(cells_energy, hitcells_ct, hitcells_E) // nowait
#pragma omp teams distribute                                                   \
    parallel for // num_teams(GRID_SIZE) num_threads(BLOCK_SIZE)
                 // //thread_limit(128) //num_teams default 1467, threads
                 // default 128
  for (int tid = 0; tid < ncells; tid++) {
    if (cells_energy[tid] > 0.) {
      unsigned int ct;
#pragma omp atomic capture
      ct = hitcells_ct[0]++;

      Cell_E ce;
      ce.cellid = tid;
      ce.energy = cells_energy[tid];
      hitcells_E[ct] = ce;
      // printf ( "ct %d %d energy %f cellid %d \n", ct, hitcells_ct[0],
      // hitcells_E[ct].energy, hitcells_E[ct].cellid);
    }
  }
  // #pragma omp taskwait
}

inline void simulate_clean(Chain0_Args &args) {

  auto cells_energy = args.cells_energy;
  auto hitcells_ct = args.hitcells_ct;

  const unsigned long ncells = args.ncells;

  int tid;
#pragma omp target is_device_ptr(cells_energy, hitcells_ct) // nowait
#pragma omp teams distribute                                                   \
    parallel for // num_teams(GRID_SIZE) num_threads(BLOCK_SIZE) // num_teams
                 // default 1467, threads default 128
  for (tid = 0; tid < ncells; tid++) {
    // printf(" num teams = %d, num threads = %d", omp_get_num_teams(),
    // omp_get_num_threads() );
    cells_energy[tid] = 0.;
    // hitcells_ct[0] = 0;
    if (tid == 0)
      hitcells_ct[tid] = 0;
  }
}

void simulate_hits(float E, int nhits, Chain0_Args &args, int select_device) {

  int m_initial_device = omp_get_initial_device();
  std::size_t m_offset = 0;

  auto t0 = std::chrono::system_clock::now();
  simulate_clean(args);
  auto t1 = std::chrono::system_clock::now();
  simulate_A(E, nhits, args);
  auto t2 = std::chrono::system_clock::now();
  simulate_ct(args);
  auto t3 = std::chrono::system_clock::now();

  int *ct = (int *)malloc(sizeof(int));
  if (omp_target_memcpy(ct, args.hitcells_ct, sizeof(int), m_offset, m_offset,
                        m_initial_device, select_device)) {
    std::cout << "ERROR: copy hitcells_ct. " << std::endl;
  }
  // gpuQ( cudaMemcpy( &ct, args.hitcells_ct, sizeof( int ),
  // cudaMemcpyDeviceToHost ) );

  if (omp_target_memcpy(args.hitcells_E_h, args.hitcells_E,
                        ct[0] * sizeof(Cell_E), m_offset, m_offset,
                        m_initial_device, select_device)) {
    std::cout << "ERROR: copy hitcells_E_h. " << std::endl;
  }
  // gpuQ( cudaMemcpy( args.hitcells_E_h, args.hitcells_E, ct * sizeof( Cell_E
  // ), cudaMemcpyDeviceToHost ) );

  auto t4 = std::chrono::system_clock::now();
  // pass result back
  args.ct = ct[0];
  //   args.hitcells_ct_h=hitcells_ct ;

#ifdef DUMP_HITCELLS
  std::cout << "hitcells: " << args.ct << "  nhits: " << nhits << "  E: " << E
            << "\n";
  std::map<unsigned int, float> cm;
  for (int i = 0; i < args.ct; ++i) {
    cm[args.hitcells_E_h[i].cellid] = args.hitcells_E_h[i].energy;
  }
  for (auto &em : cm) {
    std::cout << "  cell: " << em.first << "  " << em.second << std::endl;
  }
#endif

  timing.add(t1 - t0, t2 - t1, t3 - t2, t4 - t3);
}

} // namespace CaloGpuGeneral_omp
