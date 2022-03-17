// Copyright (C) 2002-2020 CERN for the benefit of the ATLAS collaboration

// Storage of passive histogram data used during on-device simulation.
// These properties are set by different class objects before being transferred
// to the SYCL device for processing.

#ifndef FASTCALOSYCL_SYCLCOMMON_HISTO_H_
#define FASTCALOSYCL_SYCLCOMMON_HISTO_H_
#define HIPSYCL_EXT_FP_ATOMICS
#include <CL/sycl.hpp>

namespace fastcalosycl::syclcommon {

// Holds data extracted from TFCS1DFunction objects
// for hit-to-cell assignment in accordion geometries.
// Will be retrieved from conditions database at some point.
struct Histo1DFunction {
  uint32_t max_value;      // UINT32_MAX
  unsigned int num_funcs;  // Number of functions in histogram

  // Array for bin low edge; one for each function + 1 overflow.
  // Appears that (TFCSHitCellMappingWiggle.cxx#0044) there
  // are two elements in this array.
  float* low_edge;
  // Array of sizes of functions, one for each function.
  unsigned int* sizes;
  // 1D array of pointers to histogram contents.
  uint32_t** contents;
  // 1D array of pointers to histogram borders (x-dimension), one for each
  // function.
  float** borders;
};

// Holds data extracted from TFCS2DFunctionHistogram objects
// for shape simulation.
struct Histo2DFunction {
  int num_binsx;
  int num_binsy;
  float* bordersx;
  float* bordersy;
  float* contents;
};

class Histo {
 public:
  Histo();
  Histo(cl::sycl::context* ctx);
  ~Histo();

  bool Init();

  // Loads histogram 1D function data from TObject-derived class T.
  // T should be of type TFC1DFunction and T2 a derived type of T.
  template <class T, class T2>
  bool LoadH1DF(std::vector<const T*>& func_vec,
                std::vector<float>& low_edge_vec);

  // Loads histogram 2D function data from TObject-derived class T.
  // T should be of type TFC2DFunction.
  template <class T>
  bool LoadH2DF(T& func);

  bool LoadH1DFDevice();
  bool LoadH2DFDevice();

  float RandomToH1DF(float rand, unsigned int bin);
  float RandomToH1DF(float rand, uint32_t* contents,
                                   float* borders, unsigned int num_bins,
                                   uint32_t max_value);
  void RandomToH2DF(float& val_x, float& val_y, float rand_x,
                                  float rand_y);

  inline cl::sycl::queue GetDeviceQueue() { return queue_; }

  void set_h1df(Histo1DFunction* h1df);
  void set_h1df_dev(Histo1DFunction* h1df);
  void set_h2df(Histo2DFunction* h2df);
  void set_h2df_dev(Histo2DFunction* h2df);
  Histo1DFunction* h1df();
  Histo1DFunction* h1df_dev();
  Histo2DFunction* h2df();
  Histo2DFunction* h2df_dev();

 private:
  // Finds the first index of contents whose element has value > rand_x.
  unsigned int FindIndexH1DF(uint32_t rand, unsigned int bin);
  unsigned int FindIndexH2DF(float* contents, unsigned int size,
                                           float rand_x);
  bool is_initialized_;
  cl::sycl::async_handler exception_handler;
  Histo1DFunction h1df_;  // Cached histograms; copied to 1D function device
  Histo1DFunction* h1df_ptr_;  // Pointer to host-side function histograms
  Histo1DFunction* h1df_dev_;  // Pointer to device-side 1D function histograms
  Histo2DFunction h2df_;       // Cached host-side 2D function histograms
  Histo2DFunction* h2df_ptr_;  // Pointer to a host-side 2D function histograms
  Histo2DFunction* h2df_dev_;  // Pointer to a device-side 2D function histogram

  cl::sycl::device device_;
  cl::sycl::queue queue_;   // SYCL queue; needed for context
  cl::sycl::context* ctx_;  // SYCL device context; needed for freeing memory in
                            // the destructor
};

}  // namespace fastcalosycl::syclcommon

#include "Histo.icc"

#endif  // FASTCALOSYCL_SYCLCOMMON_HISTO_H_
