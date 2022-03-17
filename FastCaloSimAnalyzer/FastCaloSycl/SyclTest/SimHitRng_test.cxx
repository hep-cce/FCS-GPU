// Copyright (C) 2002-2020 CERN for the benefit of the ATLAS collaboration

#include <SyclCommon/DeviceCommon.h>
#include <SyclRng/SimHitRng.h>

#include <CL/sycl.hpp>
#include <chrono>

static const unsigned int MAX_HITS = 100000;
static const unsigned short MAX_UNIQUE_HITS = 2000;
static const unsigned long NUM_CELLS = 200000;
static const unsigned long long SEED = 12345678987654321LLU;

// Print random numbers.
void print_random_nums(cl::sycl::queue* q, SimHitRng* rng) {
// CUDA does not support experimental::printf()
#ifndef SYCL_TARGET_CUDA
  std::cout << "Generated random numbers..." << std::endl;
  auto ev_print = q->submit([&](cl::sycl::handler& cgh) {
    cgh.parallel_for<class Dummy>(
        cl::sycl::range<1>(MAX_HITS), [=, rnd_nums_local = rng->random_nums_ptr(
                                              MAX_HITS)](cl::sycl::id<1> idx) {
          unsigned int id = (int)idx[0];
          float rnd_num = rnd_nums_local[id];
          cl::sycl::intel::experimental::printf(
              fastcalosycl::syclcommon::kTestRandomNum, id, rnd_num);
        });
  });
  ev_print.wait_and_throw();
#else
  std::cout << "CUDA does not support experimental::printf(). Cannot call "
               "print_random_nums()."
            << std::endl;
#endif
}

void test1() {
  std::cout << "test1()" << std::endl;
  SimHitRng* rng = new SimHitRng();
  if (rng) {
    if (!rng->Init(MAX_HITS, MAX_UNIQUE_HITS, NUM_CELLS, SEED)) {
      std::cout << "error in Init()!\n";
      return;
    }
    auto gentime_start = std::chrono::system_clock::now();
    rng->Generate(MAX_HITS);
    auto gentime_end = std::chrono::system_clock::now();
    std::chrono::duration<double> gentime = gentime_end - gentime_start;
    std::cout << "SimHitRng::Generate(): " << gentime.count() << std::endl;
    static cl::sycl::queue q = rng->GetQueue();
    // print_random_nums(&q, rng);
    delete (rng);
    rng = nullptr;
  }
  std::cout << "test1() OK" << std::endl;
}

int main() {
  std::cout << "SimHitRng BEGIN" << std::endl;
  test1();
  std::cout << "SimHitRng END" << std::endl;
}
