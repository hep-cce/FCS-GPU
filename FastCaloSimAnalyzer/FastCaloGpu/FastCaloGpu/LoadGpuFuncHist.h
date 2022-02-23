/*
  Copyright (C) 2002-2021 CERN for the benefit of the ATLAS collaboration
*/

#ifndef LOADGPUFUNCHIST_H
#define LOADGPUFUNCHIST_H

#include "FH_structs.h"

class LoadGpuFuncHist {

public:
  LoadGpuFuncHist();
  ~LoadGpuFuncHist();

  void  set_hf( FHs* hf_ptr ) { m_hf = hf_ptr; }
  void  set_d_hf( FHs* hf_ptr ) { m_d_hf = hf_ptr; }
  void  set_hf2d( FH2D* hf_ptr ) { m_hf2d = hf_ptr; }
  void  set_d_hf2d( FH2D* hf_ptr ) { m_d_hf2d = hf_ptr; }
  FHs*  hf() const { return m_hf; };
  FHs*  d_hf() const { return m_d_hf; };
  FH2D* hf2d() const { return m_hf2d; };
  FH2D* hf2d_d() const { return m_hf2d_d; };
  FH2D* d_hf2d() const { return m_d_hf2d; };

  void LD();
  void LD2D();

private:
  struct FHs*  m_hf{nullptr};
  struct FHs*  m_d_hf{nullptr}; // device pointer
  struct FHs*  m_hf_d{nullptr}; // host pointer to struct hold device param that is copied to device
  struct FH2D* m_hf2d{nullptr};
  struct FH2D* m_hf2d_d{nullptr}; // host poniter struct hold device param to be copied to device
  struct FH2D* m_d_hf2d{nullptr}; // device pointer
};

#endif
