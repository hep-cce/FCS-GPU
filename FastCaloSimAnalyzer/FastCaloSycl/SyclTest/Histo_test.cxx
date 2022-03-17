// Copyright (C) 2002-2020 CERN for the benefit of the ATLAS collaboration

#ifdef NDEBUG
#undef NDEBUG
#endif
#include <SyclCommon/Histo.h>
#include <assert.h>

#include <CL/sycl.hpp>
#include <iostream>

#include "ISF_FastCaloSimEvent/TFCS1DFunction.h"
#include "ISF_FastCaloSimEvent/TFCS1DFunctionInt32Histogram.h"
#include "TH1.h"
#include "TH2F.h"
#include "TRandom.h"

namespace syclcommon = fastcalosycl::syclcommon;

static const int NUM_BINS = 5;
static const float MAX_ETA = 5.0;

// Test reading of Histo1DFunctions data
// TODO: Should check all data, not just histogram data.
bool test1(std::vector<const TFCS1DFunction*>& functions,
           syclcommon::Histo* histo) {
  std::cout << "Comparing TFCS1DFunction to Histo1DFunction...";
  for (unsigned int i = 0; i < histo->h1df()->num_funcs; ++i) {
    for (unsigned int j = 0; j < histo->h1df()->sizes[i]; ++j) {
      uint32_t orig =
          ((TFCS1DFunctionInt32Histogram*)functions[i])->get_HistoContents()[j];
      uint32_t copy = histo->h1df()->contents[i][j];
      if (orig != copy) {
        return false;
      }
    }
  }
  return true;
}

// Test reading of Histo2DFunctions
bool test2() { return true; }

// main
int main() {
  std::cout << "*** Begin Histo_test ***" << std::endl;

  syclcommon::Histo* histo = new syclcommon::Histo();

  // Histo1DFunction setup
  std::vector<const TFCS1DFunction*> functions;
  std::vector<float> bin_low_edges;

  for (float eta = 0.0; eta < MAX_ETA; eta += MAX_ETA / NUM_BINS) {
    TH1* h = TFCS1DFunction::generate_histogram_random_gauss(5, 100000, -0.0125,
                                                             0.0125, 0, 0.005);
    bin_low_edges.push_back(eta);
    functions.push_back(new TFCS1DFunctionInt32Histogram(h));
    delete h;
  }
  bin_low_edges.push_back(100);

  // Load function histograms on host
  histo->LoadH1DF<TFCS1DFunction, TFCS1DFunctionInt32Histogram>(functions,
                                                                bin_low_edges);

  // test1()
  if (test1(functions, histo)) {
    std::cout << "success." << std::endl;
  } else {
    std::cout << "failed!" << std::endl;
    return -1;
  }

  // Histo2DFunction setup
  int num_binsx = 64;
  int num_binsy = 64;
  TH2F* hist = new TH2F("test2d", "test2d", num_binsx, 0, 1, num_binsy, 0, 1);
  hist->Sumw2();
  for (unsigned int ix = 1; ix <= num_binsx; ++ix) {
    for (unsigned int iy = 1; iy <= num_binsy; ++iy) {
      float random = gRandom->Rndm();
      if (random < 0.1) {
        hist->SetBinContent(ix, iy, 0);
      } else {
        hist->SetBinContent(ix, iy,
                            (0.5 + random) * (num_binsx + ix) *
                                (num_binsy * num_binsy / 2 + iy * iy));
      }
      hist->SetBinError(ix, iy, 0);
    }
  }

  // Transfer data to device
  if (!histo->Init()) {
    std::cout << "Histo::Init() failed!" << std::endl;
    return -1;
  }
  if (!histo->LoadH1DFDevice()) {
    std::cout << "Histo::LoadH1DFDevice() failed!" << std::endl;
  }
  cl::sycl::queue q = histo->GetDeviceQueue();

#ifndef SYCL_TARGET_CUDA
  std::cout << "Test device histogram data..." << std::endl;
  q.submit([&](cl::sycl::handler& cgh) {
     cgh.parallel_for<class Dummy>(
         cl::sycl::range<1>(histo->h1df()->num_funcs),
         [=, dev_funcs = histo->h1df_dev()](cl::sycl::id<1> idx) {
           unsigned int id = (int)idx[0];
           cl::sycl::intel::experimental::printf(
               fastcalosycl::syclcommon::kHistoPrintFuncNum, id);
           unsigned int size = dev_funcs->sizes[id];
           uint32_t* contents = dev_funcs->contents[id];
           for (unsigned int i = 0; i < size; ++i) {
             cl::sycl::intel::experimental::printf(
                 fastcalosycl::syclcommon::kHistoPrintContents, id, i,
                 (unsigned long)contents[i]);
           }
         });
   }).wait_and_throw();
#else
  std::cout << "CUDA does not support experimental::printf(). Cannot call
               "
               "test_cells()."
            << std::endl;
#endif

  if (histo) {
    delete histo;
    histo = nullptr;
  }

  std::cout << "*** End Histo_test ***" << std::endl;
  return 0;
}  // main
