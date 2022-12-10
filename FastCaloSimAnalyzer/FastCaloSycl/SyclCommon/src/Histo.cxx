// Copyright (C) 2002-2020 CERN for the benefit of the ATLAS collaboration

#include <SyclCommon/Histo.h>

namespace fastcalosycl::syclcommon {

Histo::Histo()
    : is_initialized_(false),
      h1df_ptr_(nullptr),
      h1df_dev_(nullptr),
      h2df_ptr_(nullptr),
      h2df_dev_(nullptr) {}

Histo::Histo(sycl::context* ctx)
    : is_initialized_(false),
      h1df_ptr_(nullptr),
      h1df_dev_(nullptr),
      h2df_ptr_(nullptr),
      h2df_dev_(nullptr),
      ctx_(ctx) {}

Histo::~Histo() {
  if (h1df_dev_) {
    // for (unsigned int i = 0; i < h1df_.num_funcs; i++) {
    // sycl::free(h1df_dev_->contents[i], *ctx_);
    // sycl::free(h1df_dev_->borders[i], *ctx_);
    // }
    // sycl::free(h1df_dev_->low_edge, *ctx_);
    // sycl::free(h1df_dev_->sizes, *ctx_);
    // sycl::free(h1df_dev_->contents, *ctx_);
    // sycl::free(h1df_dev_->borders, *ctx_);
    // sycl::free(h1df_dev_, *ctx_);
    sycl::free(h1df_dev_, *ctx_);
    free(h1df_.sizes);
    free(h1df_.contents);
    free(h1df_.borders);
  }

  if (h2df_dev_) {
    // sycl::free(h2df_dev_->bordersx, *ctx_);
    // sycl::free(h2df_dev_->bordersy, *ctx_);
    // sycl::free(h2df_dev_->contents, *ctx_);
    sycl::free(h2df_dev_, *ctx_);
  }
}

}  // namespace fastcalosycl::syclcommon
