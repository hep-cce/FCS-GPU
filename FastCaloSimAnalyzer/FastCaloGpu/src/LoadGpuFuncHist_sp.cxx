/*
  Copyright (C) 2002-2021 CERN for the benefit of the ATLAS collaboration
*/

#include "LoadGpuFuncHist.h"
#include <iostream>
#include <cstring>


LoadGpuFuncHist::LoadGpuFuncHist() {
  // std::cout << "============= LoadGpuFuncHist ================\n";
}

LoadGpuFuncHist::~LoadGpuFuncHist() {
  delete m_hf2d_h->h_bordersx ;
  delete m_hf2d_h->h_bordersy ;
  delete m_hf2d_h->h_contents;
  delete m_hf2d;

  delete m_hf_h->low_edge;
  delete m_hf_h->h_szs;
  delete m_hf_h->d_contents1D;
  delete m_hf_h->d_borders1D;
  delete m_hf_h;
}

void LoadGpuFuncHist::LD2D() {
  if ( !m_hf2d ) {
    std::cout << "Error Load 2DFunctionHisto " << std::endl;
    return;
  }

  FH2D* hf_ptr = new FH2D;

  hf_ptr->nbinsx = ( *m_hf2d ).nbinsx;
  hf_ptr->nbinsy = ( *m_hf2d ).nbinsy;

  // FIXME!! This should be done by making TFCSHistoLateralShapeParametrization::m_hist a ptr
  float *bx = new float[hf_ptr->nbinsx+1];
  float *by = new float[hf_ptr->nbinsy+1];
  float *ct = new float[hf_ptr->nbinsx * hf_ptr->nbinsy];

  std::memcpy(bx, m_hf2d->h_bordersx, sizeof(float)*(hf_ptr->nbinsx+1));
  std::memcpy(by, m_hf2d->h_bordersy, sizeof(float)*(hf_ptr->nbinsy+1));
  std::memcpy(ct, m_hf2d->h_contents, sizeof(float)*hf_ptr->nbinsy*hf_ptr->nbinsx);

  hf_ptr->h_bordersx = bx;
  hf_ptr->h_bordersy = by;
  hf_ptr->h_contents = ct;
  
  m_hf2d_h = hf_ptr;
  m_hf2d_d = hf_ptr;

}

void LoadGpuFuncHist::LD() {
  // this call  assume  already have Histofuncs set in m_hf
  // this function allocate memory of GPU and deep copy m_hf to m_hf_d

  if ( !m_hf ) {
    std::cout << "Error Load WiggleHistoFunctions " << std::endl;
    return;
  }

  FHs* hf_ptr = new FHs;

  hf_ptr->s_MaxValue       = ( *m_hf ).s_MaxValue;
  hf_ptr->nhist            = ( *m_hf ).nhist;
  hf_ptr->mxsz             = ( *m_hf ).mxsz;

  // FIXME:: should do this at level of m_hf
  // hf_ptr->low_edge = ( *m_hf ).low_edge;
  // hf_ptr->h_szs    = ( *m_hf ).h_szs;

  float* le = new float[hf_ptr->nhist+1];
  unsigned int *hs = new unsigned int[hf_ptr->nhist];

  std::memcpy(le, m_hf->low_edge, sizeof(float)*(hf_ptr->nhist+1));
  std::memcpy(hs, m_hf->h_szs, sizeof(unsigned int)*hf_ptr->nhist);

  hf_ptr->low_edge = le;
  hf_ptr->h_szs = hs;
  

  // std::cout << "low edge:\n";
  // for (int i=0; i<hf_ptr->nhist; ++i) {
  //   std::cout << " " << i << " " << hf_ptr->low_edge[i] << "\n";
  // }
    
  hf_ptr->h_contents = ( *m_hf ).h_contents;
  hf_ptr->h_borders  = ( *m_hf ).h_borders;

  hf_ptr->d_contents1D = (uint32_t*) malloc(hf_ptr->nhist * hf_ptr->mxsz * sizeof(uint32_t));
  hf_ptr->d_borders1D  = (float*)    malloc(hf_ptr->nhist * hf_ptr->mxsz * sizeof(float));
  
  for ( size_t i = 0; i < hf_ptr->nhist; ++i ) {
    memcpy( &hf_ptr->d_contents1D[i*hf_ptr->mxsz], (*m_hf).h_contents[i],
            (*m_hf).h_szs[i] * sizeof(uint32_t));

    memcpy( &hf_ptr->d_borders1D[i*hf_ptr->mxsz], (*m_hf).h_borders[i],
            (*m_hf).h_szs[i] * sizeof( float ));
    
  }

  // std::cout << "contents\n";
  // for (int i=0; i<hf_ptr->nhist*hf_ptr->mxsz; ++i) {
  //   std::cout << " " << i << " " << hf_ptr->d_borders1D[i] << std::endl;
  // }


  m_hf_h      = hf_ptr;
  m_hf_d       = hf_ptr;
  

}
