/*
  Copyright (C) 2002-2021 CERN for the benefit of the ATLAS collaboration
*/

#include "LoadGpuFuncHist.h"
#include <iostream>


LoadGpuFuncHist::LoadGpuFuncHist() {
  std::cout << "============= LoadGpuFuncHist ================\n";
}

LoadGpuFuncHist::~LoadGpuFuncHist() {
}

void LoadGpuFuncHist::LD2D() {
  if ( !m_hf2d ) {
    std::cout << "Error Load 2DFunctionHisto " << std::endl;
    return;
  }

  FH2D* hf_ptr = new FH2D;

  hf_ptr->nbinsx = ( *m_hf2d ).nbinsx;
  hf_ptr->nbinsy = ( *m_hf2d ).nbinsy;

  hf_ptr->h_bordersx = (*m_hf2d).h_bordersx;
  hf_ptr->h_bordersy = (*m_hf2d).h_bordersx;
  hf_ptr->h_contents = (*m_hf2d).h_contents;
  
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

  hf_ptr->low_edge = ( *m_hf ).low_edge;
  hf_ptr->h_szs    = ( *m_hf ).h_szs;

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
