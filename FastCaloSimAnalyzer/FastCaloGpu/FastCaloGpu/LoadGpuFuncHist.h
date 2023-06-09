/*
  Copyright (C) 2002-2021 CERN for the benefit of the ATLAS collaboration
*/

#ifndef LOADGPUFUNCHIST_H
#define LOADGPUFUNCHIST_H

#include "FH_structs.h"
#include <omp.h>

#ifdef USE_KOKKOS
#  include "FH_views.h"
#endif

class LoadGpuFuncHist {

public:
  LoadGpuFuncHist();
  ~LoadGpuFuncHist();

  void  set_hf( FHs* hf_ptr ) { m_hf = hf_ptr; }
  void  set_d_hf( FHs* hf_ptr ) { m_hf_d = hf_ptr; }
  void  set_hf2d( FH2D* hf_ptr ) { m_hf2d = hf_ptr; }
  void  set_d_hf2d( FH2D* hf_ptr ) { m_hf2d_d = hf_ptr; }
  FHs*  hf() const { return m_hf; }
  FHs*  hf_h() const { return m_hf_h; } // device ptr on HOST
  FHs*  hf_d() const { return m_hf_d; } // device ptr on DEVICE
  FH2D* hf2d() const { return m_hf2d; }
  FH2D* hf2d_h() const { return m_hf2d_h; } // device ptr on HOST
  FH2D* hf2d_d() const { return m_hf2d_d; } // device ptr on DEVICE

  void LD();
  void LD2D();

  void select_omp_device ( ) {
  if ( offload_var == "mandatory" ) 
	m_select_device = m_default_device;
  else if ( offload_var == "disabled" )
      m_select_device = m_initial_device;
  };

private:
  struct FHs* m_hf{0};
  struct FHs* m_hf_d{0}; // device pointer
  struct FHs* m_hf_h{0}; // host pointer to struct hold device param that is copied to device

  struct FH2D* m_hf2d{0};
  struct FH2D* m_hf2d_d{0}; // device pointer
  struct FH2D* m_hf2d_h{0}; // host pointer struct hold device param to be copied to device

  //TODO: Wrap device IDs from omp APIs to a class
  //link to GeoLoadGpu.h, Rand4Hits.h, LoadGpuFuncHist
#ifdef USE_OMPGPU
  int m_num_devices    = omp_get_num_devices();
  int m_default_device = omp_get_default_device();
  int m_initial_device = omp_get_initial_device();
  std::size_t m_offset = 0;
  const char *env_var = "OMP_TARGET_OFFLOAD";
  std::string offload_var = std::getenv (env_var);
  int m_select_device = m_default_device;
#endif


#ifdef USE_KOKKOS
  FH2D_v*            m_hf2d_v{0};
  Kokkos::View<FH2D> m_hf2d_dv;

  FHs_v*            m_hf_v{0};
  Kokkos::View<FHs> m_hf_dv;
#endif
};

#endif
