// Copyright (C) 2002-2020 CERN for the benefit of the ATLAS collaboration

#include <SyclCommon/DeviceCommon.h>
#include <SyclGeo/GeoRegion.h>

#include <algorithm>

GeoRegion::GeoRegion()
    : cell_grid_(nullptr),
      cell_grid_device_(nullptr),
      cells_(nullptr),
      cells_device_(nullptr),
      index_(0),
      cell_grid_eta_(0),
      cell_grid_phi_(0),
      xy_grid_adjust_(0.0),
      deta_(0.0),
      dphi_(0.0),
      min_eta_(0.0),
      min_phi_(0.0),
      max_eta_(0.0),
      max_phi_(0.0),
      min_eta_raw_(0.0),
      min_phi_raw_(0.0),
      max_eta_raw_(0.0),
      max_phi_raw_(0.0),
      eta_corr_(0.0),
      phi_corr_(0.0),
      min_eta_corr_(0.0),
      max_eta_corr_(0.0),
      min_phi_corr_(0.0),
      max_phi_corr_(0.0) {}
