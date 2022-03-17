// Copyright (C) 2002-2020 CERN for the benefit of the ATLAS collaboration

#include <SyclRng/SimHitRng.h>

SimHitRng::SimHitRng()
    : engine_(nullptr),
      dist_(nullptr),
      allocd_num_hits_(0),
      current_num_hits_(0),
      random_nums_(nullptr),
      cells_energy_(nullptr),
      cell_props_(nullptr),
      num_unique_hits_(nullptr),
      is_initialized_(false),
      ctx_(nullptr) {}

SimHitRng::SimHitRng(cl::sycl::context* ctx)
    : engine_(nullptr),
      dist_(nullptr),
      allocd_num_hits_(0),
      current_num_hits_(0),
      random_nums_(nullptr),
      cells_energy_(nullptr),
      cell_props_(nullptr),
      num_unique_hits_(nullptr),
      is_initialized_(false),
      ctx_(ctx) {}

SimHitRng::~SimHitRng() {
  if (is_initialized_) {
    Dealloc();
  }
}

bool SimHitRng::Init(unsigned int max_nhits, unsigned short max_unique_hits,
                     unsigned long ncells, unsigned long long seed) {
  // Initialize device, queue and context
  if (!ctx_) {
    device_ = fastcalosycl::syclcommon::GetTargetDevice();
    queue_ = cl::sycl::queue(device_);
    ctx_ = new cl::sycl::context(queue_.get_context());
  } else {
    device_ = ctx_->get_devices()[0];
    queue_ = cl::sycl::queue(*ctx_, device_);
  }
  allocd_num_hits_ = max_nhits;

  // Name of the device running on
  std::string dev_name = device_.get_info<cl::sycl::info::device::name>();
  std::cout << "SyclRng::SimHitRng  Using device \"" << dev_name << "\""
            << std::endl;

  // Allocate memory
  if (!Alloc(ncells, max_unique_hits)) {
    std::cout << "SimHitRng::Init() failed!\n";
    return false;
  }

  // Memory allocated successfully.
  std::cout << "Device random number memory allocated..." << std::endl;
  std::cout << "\tallocd_num_hits: " << allocd_num_hits_ << std::endl
            << "\tsize: " << allocd_num_hits_ * sizeof(unsigned int) / 1000
            << " kb\n"
            << std::endl;

  // Initialize engine and distribution for RNG
  engine_ = new philo_engine_t(queue_, seed);
  dist_ = new uniform_dist_t(0.0f, 1.0f);

  return true;
}

cl::sycl::context* SimHitRng::GetContext() { return ctx_; }

cl::sycl::queue& SimHitRng::GetQueue() { return queue_; }

void SimHitRng::Generate(unsigned int nhits) {
  oneapi::mkl::rng::generate(*dist_, *engine_, nhits, random_nums_).wait();
}

bool SimHitRng::Alloc(unsigned long ncells, unsigned short max_unique_hits) {
  // Don't reallocated if memory is already allocated.
  if (is_initialized_) {
    return true;
  }

  // Allocate device-side memory for random numbers array.
  float* rn = (float*)malloc_device(allocd_num_hits_ * sizeof(unsigned int),
                                    device_, *ctx_);
  if (!rn) {
    std::cout << "Cannot allocate device-side memory for random numbers!"
              << std::endl;
    return false;
  }
  random_nums_ = rn;

  // Allocate device-side memory for energy in cells
  float* ce = (float*)malloc_device(ncells * sizeof(float), device_, *ctx_);
  if (!ce) {
    std::cout << "Cannot allocate device-side memory for cell energy!"
              << std::endl;
    return false;
  }
  cells_energy_ = ce;

  // Allocate device-side memory for cell properties
  fastcalosycl::syclcommon::CellProps* cp =
      (fastcalosycl::syclcommon::CellProps*)malloc_device(
          max_unique_hits * sizeof(fastcalosycl::syclcommon::CellProps),
          device_, *ctx_);
  if (!cp) {
    std::cout << "Cannot allocate device-side memory for cell properties!"
              << std::endl;
    return false;
  }
  cell_props_ = cp;

  // Allocate device-side memory for unique hits counter
  int* uh = (int*)malloc_device(sizeof(int), device_, *ctx_);
  if (!uh) {
    std::cout << "Cannot allocate device-side memory for unique hits counter!"
              << std::endl;
    return false;
  }
  num_unique_hits_ = uh;

  is_initialized_ = true;
  return true;
}

void SimHitRng::Dealloc() {
  if (is_initialized_) {
    cl::sycl::free(random_nums_, *ctx_);
    cl::sycl::free(cells_energy_, *ctx_);
    cl::sycl::free(cell_props_, *ctx_);
    cl::sycl::free(num_unique_hits_, *ctx_);
    delete (dist_);
    delete (engine_);
    engine_ = nullptr;
    dist_ = nullptr;
    is_initialized_ = false;
  }
}

// void SimHitRng::add_current_hits(unsigned int nhits) {
//   if (over_alloc(nhits)) {
//     current_num_hits_ = nhits;
//   } else {
//     current_num_hits_ += nhits;
//   }
// }

void SimHitRng::set_genevent(genevent_t /*gen*/) {}

void SimHitRng::set_random_nums(float* rn) { random_nums_ = rn; }

void SimHitRng::set_allocd_num_hits(unsigned int nhits) {
  allocd_num_hits_ = nhits;
}

void SimHitRng::set_current_num_hits(unsigned int nhits) {
  current_num_hits_ = nhits;
}

unsigned int SimHitRng::get_allocd_num_hits() { return allocd_num_hits_; }

unsigned int SimHitRng::get_current_num_hits() { return current_num_hits_; }

float* SimHitRng::random_nums_ptr(unsigned int nhits) {
  // Check if we need more memory
  if (over_alloc(nhits)) {
    // Regenerate nhits random numbers
    Generate(nhits);
    return random_nums_;
  } else {
    float* rn_ptr = &(random_nums_[current_num_hits_]);
    return rn_ptr;
  }
}

int* SimHitRng::get_num_unique_hits() { return num_unique_hits_; }

float* SimHitRng::get_cells_energy() { return cells_energy_; }

genevent_t SimHitRng::get_genevent() { return genevent_; }

bool SimHitRng::over_alloc(unsigned int nhits) {
  return (current_num_hits_ + nhits) > allocd_num_hits_;
}
