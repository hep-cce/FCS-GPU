// Copyright (C) 2002-2020 CERN for the benefit of the ATLAS collaboration

#ifndef FASTCALOSYCL_SYCLCOMMON_DEVICECOMMON_H_
#define FASTCALOSYCL_SYCLCOMMON_DEVICECOMMON_H_
#define HIPSYCL_EXT_FP_ATOMICS
#include <CL/sycl.hpp>

namespace fastcalosycl::syclcommon {

#ifdef SYCL_TARGET_CUDA
class CUDASelector : public cl::sycl::device_selector {
 public:
  int operator()(const cl::sycl::device& device) const override {
    const std::string device_vendor = device.get_info<cl::sycl::info::device::vendor>();
    const std::string device_driver =
        device.get_info<cl::sycl::info::device::driver_version>();

    if (device.is_gpu() &&
        (device_vendor.find("NVIDIA") != std::string::npos) &&
        (device_driver.find("CUDA") != std::string::npos)) {
      return 1;
    };
    return -1;
  }
};
#endif
#ifdef SYCL_TARGET_HIP
class AMDSelector : public cl::sycl::device_selector {
 public:
  int operator()(const cl::sycl::device& device) const override {
    const std::string device_vendor = device.get_info<cl::sycl::info::device::vendor>();
    const std::string device_driver =
        device.get_info<cl::sycl::info::device::driver_version>();
    const std::string device_name = device.get_info<cl::sycl::info::device::name>();

    if (device.is_gpu() && (device_vendor.find("AMD") != std::string::npos)) {
      return 1;
    }
    return -1;
  }
};
#endif

// Defines for printing within command group scope, i.e. SYCL kernel.
#ifdef __SYCL_DEVICE_ONLY__
#define CONSTANT __attribute__((opencl_constant))
#else
#define CONSTANT
#endif

// Print function for device-side queries, mostly for testing and debugging.
// static const CONSTANT char Print*;

// Strings
static const CONSTANT char kString[] = "%s\n";
// Floats
static const CONSTANT char kFloat[] = "-- %f\n";
// IDs
static const CONSTANT char kLocalId[] = "local_id[%lu]\n";
static const CONSTANT char kGroup[] = "group[%lu]\n";
static const CONSTANT char kLocalRange[] = "local_range[%lu]\n";
static const CONSTANT char kIds0[] =
    "group0[%lu], local_range0[%lu], local_id0[%lu] \n";
static const CONSTANT char kIds1[] =
    "group1[%lu], local_range1[%lu], local_id1[%lu] \n";
static const CONSTANT char kWid0[] = "wid0[%lu]\n";
static const CONSTANT char kWid1[] = "wid1[%lu]\n";
static const CONSTANT char kWid[] = "wid[%lu]\n";

// Print cell info
static const CONSTANT char kCellInfo[] =
    "  device_cell :: id [%llx], hash_id [%d]\n";

// Print device info
static const CONSTANT char kDeviceGeoInfo[] =
    "  DeviceGeo :: nregions [%u], ncells [%lu]\n";

// Print region info
static const CONSTANT char kRegionInfo[] =
    "  device_region :: neta [%d], nphi [%d], index [%llu] \n";

// Test integer print
static const CONSTANT char kTestInt[] = "  test_int :: int [%d]\n";
static const CONSTANT char kTestCellEnergy[] = "  cell[%d] :: E [%f]\n";

// SimEvent tests
static const CONSTANT char kPrintRandoms2[] = "  rnd1[%f], rnd2[%f]\n";
static const CONSTANT char kPrintEtaPhi[] = "  eta[%f], phi[%f]\n";
static const CONSTANT char kPrintNEtaNPhi[] = "  eta[%d], phi[%d]\n";
static const CONSTANT char kPrintIdEtaPhi[] = "  dde[%lld], eta[%f], phi[%f]\n";

// Print random numbers
static const CONSTANT char kTestRandomNum[] = "  test_random_num[%d] = [%f]\n";

// SyclCommon::Histo
static const CONSTANT char kHistoPrintFuncNum[] = "f[%d]\n";
static const CONSTANT char kHistoPrintContents[] = "  contents[%d][%d]: %lu\n";
static const CONSTANT char kHistoPrintBorders[] = "  borders[%d][%d]: %f\n";

// Gets the target device, as defined in the cmake configuration.
static inline cl::sycl::device GetTargetDevice() {
  cl::sycl::device dev;
#if defined SYCL_TARGET_CUDA
  CUDASelector cuda_selector;
  try {
    dev = cl::sycl::device(cuda_selector);
  } catch (...) {
  }
#elif defined SYCL_TARGET_HIP
  AMDSelector selector;
  try {
    dev = cl::sycl::device(selector);
  } catch (...) {
  }
#elif defined SYCL_TARGET_DEFAULT
  dev = cl::sycl::device(cl::sycl::default_selector());
#elif defined SYCL_TARGET_CPU
  dev = cl::sycl::device(cl::sycl::cpu_selector());
#elif defined SYCL_TARGET_GPU
  dev = cl::sycl::device(cl::sycl::gpu_selector());
#else
  dev = cl::sycl::device(cl::sycl::host_selector());
#endif

  return dev;
}

static inline cl::sycl::context GetSharedContext() {
  cl::sycl::platform platform;
#if defined SYCL_TARGET_CUDA
  CUDASelector cuda_selector;
  try {
    platform = cl::sycl::platform(cuda_selector);
  } catch (...) {
  }
#elif defined SYCL_TARGET_DEFAULT
  platform = cl::sycl::platform(cl::sycl::default_selector());
#elif defined SYCL_TARGET_CPU
  platform = cl::sycl::platform(cl::sycl::cpu_selector());
#elif defined SYCL_TARGET_GPU
  platform = cl::sycl::platform(cl::sycl::gpu_selector());
#else
  platform = cl::sycl::platform(cl::sycl::host_selector());

#endif

  return cl::sycl::context(platform);
}

}  // namespace fastcalosycl::syclcommon

#endif  // FASTCALOSYCL_SYCLCOMMON_DEVICECOMMON_H_