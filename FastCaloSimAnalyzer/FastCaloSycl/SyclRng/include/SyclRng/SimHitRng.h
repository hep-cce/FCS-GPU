// Copyright (C) 2002-2020 CERN for the benefit of the ATLAS collaboration

#ifndef FASTCALOSYCL_SYCLRNG_SIMHITRNG_H_
#define FASTCALOSYCL_SYCLRNG_SIMHITRNG_H_

#include <SyclCommon/Props.h>
#define HIPSYCL_EXT_FP_ATOMICS
#include <CL/sycl.hpp>
#include <oneapi/mkl.hpp>
//#include "mkl_rng_sycl.hpp"

typedef cl::sycl::event genevent_t;
typedef oneapi::mkl::rng::philox4x32x10 philo_engine_t;
typedef oneapi::mkl::rng::uniform<float> uniform_dist_t;

// Random number generator class for simulation hits.
// Random numbers generated by sampling uniform distributions.
class SimHitRng {
 public:
  SimHitRng();
  SimHitRng(cl::sycl::context* ctx);
  ~SimHitRng();

  // Initialize SYCL queue, context and device, and allocate device memory for
  // random numbers.
  bool Init(unsigned int max_nhits, unsigned short max_unique_hits,
            unsigned long ncells, unsigned long long seed);

  // Retrieve the associated cl::sycl::queue.
  // All device-side data management is done through the returned object.
  cl::sycl::queue& GetQueue();

  // Performs random number generation.
  void Generate(unsigned int nhits);

  // Increments the number of current hits. If nhits is larger than
  // current_num_hits_, the latter is reassigned to the former.
  inline void add_current_hits(unsigned int nhits) {
    if (over_alloc(nhits)) {
        current_num_hits_ = nhits;
    } else {
      current_num_hits_ += nhits;
    }
  }

  // Sets the generator event.
  void set_genevent(genevent_t gen);

  // Sets the random numbers pointer.
  void set_random_nums(float* rn);

  // Sets the allocated number of hits.
  void set_allocd_num_hits(unsigned int nhits);

  // Sets the current number of hits.
  void set_current_num_hits(unsigned int nhits);

  unsigned int get_allocd_num_hits();
  unsigned int get_current_num_hits();
  int* get_num_unique_hits();
  float* random_nums_ptr(unsigned int nhits);
  float* get_cells_energy();
  genevent_t get_genevent();

  cl::sycl::context* GetContext();

 private:
  // Determines if nhits > current_num_hits_.
  bool over_alloc(unsigned int nhits);

  // Allocates memory for cells' energy deposits.
  bool Alloc(unsigned long ncells, unsigned short max_unique_hits);

  // Deallocates memory; called in dtor.
  void Dealloc();

  genevent_t genevent_;            // RNG event; runs generation
  philo_engine_t* engine_;         // RNG engine
  uniform_dist_t* dist_;           // Distribution from which to sample
  unsigned int allocd_num_hits_;   // Total memory allocated for hits
  unsigned int current_num_hits_;  // Current number of hits
  float* random_nums_;             // Array of random numbers
  float* cells_energy_;            // Array of energy deposits in cells
  fastcalosycl::syclcommon::CellProps*
      cell_props_;           // Array of CellProps structs
  int* num_unique_hits_;     // Pointer to single int value containing number of
                             // unique cell hits
  bool is_initialized_;      // Has Init() been called, and return true
  cl::sycl::device device_;  // SYCL device
  cl::sycl::queue queue_;    // SYCL queue; needed for context
  cl::sycl::context*         // SYCL context; needed for freeing device memory
      ctx_;  // SYCL device context; needed for freeing memory in the destructor
};

#endif  // FASTCALOSYCL_SYCLRNG_SIMHITRNG_H_
