/*
  Copyright (C) 2002-2021 CERN for the benefit of the ATLAS collaboration
*/

#include "LoadGpuFuncHist.h"
#include <iostream>
#include <cstring>

LoadGpuFuncHist::LoadGpuFuncHist() {
  // std::cout << "============= LoadGpuFuncHist ================\n";
}

/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

LoadGpuFuncHist::~LoadGpuFuncHist() {
  free( m_hf );
  free( m_hf_d );

  free( m_hf2d );
  free( m_hf2d_d );
}

/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

void LoadGpuFuncHist::LD2D() {
  if ( !m_hf2d ) {
    std::cout << "Error Load 2DFunctionHisto " << std::endl;
    return;
  }

  FH2D* hf_ptr = new FH2D;
  FH2D  hf;

  hf.nbinsx = ( *m_hf2d ).nbinsx;
  hf.nbinsy = ( *m_hf2d ).nbinsy;

  hf.h_bordersx = new float[hf.nbinsx + 1];
  hf.h_bordersy = new float[hf.nbinsy + 1];
  hf.h_contents = new float[hf.nbinsy * hf.nbinsx];

  std::memcpy( hf.h_bordersx, ( *m_hf2d ).h_bordersx, ( hf.nbinsx + 1 ) * sizeof( float ) );
  std::memcpy( hf.h_bordersy, ( *m_hf2d ).h_bordersy, ( hf.nbinsy + 1 ) * sizeof( float ) );
  std::memcpy( hf.h_contents, ( *m_hf2d ).h_contents, ( hf.nbinsx * hf.nbinsy ) * sizeof( float ) );

  *( hf_ptr ) = hf;
  m_hf2d_d    = hf_ptr;

  m_d_hf2d = m_hf2d_d;
}

/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

void LoadGpuFuncHist::LD() {
  // this call  assume  already have Histofuncs set in m_hf
  // this function allocate memory of GPU and deep copy m_hf to m_hf_d

  if ( !m_hf ) {
    std::cout << "Error Load WiggleHistoFunctions " << std::endl;
    return;
  }

  FHs* hf_ptr         = new FHs;
  hf_ptr->s_MaxValue  = ( *m_hf ).s_MaxValue;
  hf_ptr->nhist       = ( *m_hf ).nhist;
  unsigned int* h_szs = ( *m_hf ).h_szs; // already allocateded on host ;

  hf_ptr->low_edge = new float[hf_ptr->nhist + 1];
  std::memcpy( hf_ptr->low_edge, ( *m_hf ).low_edge, ( hf_ptr->nhist + 1 ) * sizeof( float ) );

  hf_ptr->h_szs = new unsigned int[hf_ptr->nhist];
  std::memcpy( hf_ptr->h_szs, ( *m_hf ).h_szs, hf_ptr->nhist * sizeof( unsigned int ) );

  hf_ptr->h_contents = (uint32_t**) malloc( hf_ptr->nhist * sizeof( uint32_t* ) );
  hf_ptr->h_borders  = (float**)    malloc( hf_ptr->nhist * sizeof( float* ) );

  uint32_t** contents_ptr = (uint32_t**)malloc( hf_ptr->nhist * sizeof( uint32_t* ) );
  float**    borders_ptr  = (float**)malloc( hf_ptr->nhist * sizeof( float* ) );

  for ( unsigned int i = 0; i < hf_ptr->nhist; ++i ) {

    contents_ptr[i] = new uint32_t[h_szs[i]];
    borders_ptr[i]  = new float[h_szs[i] + 1];

    std::memcpy( contents_ptr[i], ( *m_hf ).h_contents[i], h_szs[i] * sizeof( uint32_t ) );
    std::memcpy( borders_ptr[i], ( *m_hf ).h_borders[i], ( h_szs[i] + 1 ) * sizeof( float ) );
  }

  std::memcpy( hf_ptr->h_contents, contents_ptr, hf_ptr->nhist * sizeof( uint32_t* ) );
  std::memcpy( hf_ptr->h_borders, borders_ptr, hf_ptr->nhist * sizeof( float* ) );

  m_d_hf = hf_ptr;
  m_hf_d = hf_ptr;

  // FHs* hf_ptr = new FHs;

  // hf_ptr->s_MaxValue       = ( *m_hf ).s_MaxValue;
  // hf_ptr->nhist            = ( *m_hf ).nhist;
  // hf_ptr->mxsz             = ( *m_hf ).mxsz;

  // // FIXME:: should do this at level of m_hf
  // // hf_ptr->low_edge = ( *m_hf ).low_edge;
  // // hf_ptr->h_szs    = ( *m_hf ).h_szs;

  // float* le = new float[hf_ptr->nhist+1];
  // unsigned int *hs = new unsigned int[hf_ptr->nhist];

  // std::memcpy(le, m_hf->low_edge, sizeof(float)*(hf_ptr->nhist+1));
  // std::memcpy(hs, m_hf->h_szs, sizeof(unsigned int)*hf_ptr->nhist);

  // hf_ptr->low_edge = le;
  // hf_ptr->h_szs = hs;

  // hf_ptr->h_contents = ( *m_hf ).h_contents;
  // hf_ptr->h_borders  = ( *m_hf ).h_borders;

  // hf_ptr->d_contents1D = (uint32_t*) malloc(hf_ptr->nhist * hf_ptr->mxsz * sizeof(uint32_t));
  // hf_ptr->d_borders1D  = (float*)    malloc(hf_ptr->nhist * hf_ptr->mxsz * sizeof(float));

  // for ( size_t i = 0; i < hf_ptr->nhist; ++i ) {
  //   memcpy( &hf_ptr->d_contents1D[i*hf_ptr->mxsz], (*m_hf).h_contents[i],
  //           (*m_hf).h_szs[i] * sizeof(uint32_t));

  //   memcpy( &hf_ptr->d_borders1D[i*hf_ptr->mxsz], (*m_hf).h_borders[i],
  //           (*m_hf).h_szs[i] * sizeof( float ));

  // }

  // m_hf_h      = hf_ptr;
  // m_hf_d       = hf_ptr;
}
