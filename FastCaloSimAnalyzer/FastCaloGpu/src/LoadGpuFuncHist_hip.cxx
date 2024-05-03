/*
  Copyright (C) 2002-2021 CERN for the benefit of the ATLAS collaboration
*/

#include "LoadGpuFuncHist.h"
#include <iostream>
#include "gpuQ.h"

LoadGpuFuncHist::LoadGpuFuncHist() {}

  LoadGpuFuncHist::~LoadGpuFuncHist() {
  free( m_hf );
  auto err = hipFree( ( *m_hf_h ).low_edge );
  err = hipFree( ( *m_hf_h ).h_szs );
  err = hipFree( ( *m_hf_h ).h_contents );
  err = hipFree( ( *m_hf_h ).h_borders );
  err = hipFree( ( *m_hf_h ).d_contents1D );
  err = hipFree( ( *m_hf_h ).d_borders1D );
  free( m_hf_h );
  err = hipFree( m_hf_d );

  free( m_hf2d );
  err = hipFree( ( *m_hf2d_h ).h_bordersx );
  err = hipFree( ( *m_hf2d_h ).h_bordersy );
  err = hipFree( ( *m_hf2d_h ).h_contents );
  free( m_hf2d_h );
  err = hipFree( m_hf2d_d );
}

void LoadGpuFuncHist::LD2D() {
  if ( !m_hf2d ) {
    std::cout << "Error Load 2DFunctionHisto " << std::endl;
    return;
  }

  FH2D* hf_ptr = new FH2D;
  FH2D  hf;

  hf.nbinsx = ( *m_hf2d ).nbinsx;
  hf.nbinsy = ( *m_hf2d ).nbinsy;
  // std::cout << ".....Loading  2DFnctionHists, Size of hist" <<  hf.nbinsx  << "x" << hf.nbinsy << std::endl ;
  // std::cout << "(*m_hf2d).h_bordersy, pointer " <<  (*m_hf2d).h_bordersy   << std::endl ;

  gpuQ( hipMalloc( (void**)&hf.h_bordersx, ( hf.nbinsx + 1 ) * sizeof( float ) ) );
  gpuQ( hipMalloc( (void**)&hf.h_bordersy, ( hf.nbinsy + 1 ) * sizeof( float ) ) );
  gpuQ( hipMalloc( (void**)&hf.h_contents, ( hf.nbinsy * hf.nbinsx ) * sizeof( float ) ) );
  gpuQ( hipMemcpy( hf.h_bordersx, ( *m_hf2d ).h_bordersx, ( hf.nbinsx + 1 ) * sizeof( float ),
                    hipMemcpyHostToDevice ) );
  //  std::cout << "hf.h_bordersy, pointer " <<  hf.h_bordersy  <<  std::endl ;
  gpuQ( hipMemcpy( hf.h_bordersy, ( *m_hf2d ).h_bordersy, ( hf.nbinsy + 1 ) * sizeof( float ),
                    hipMemcpyHostToDevice ) );
  gpuQ( hipMemcpy( hf.h_contents, ( *m_hf2d ).h_contents, ( hf.nbinsx * hf.nbinsy ) * sizeof( float ),
                    hipMemcpyHostToDevice ) );
  
  *( hf_ptr ) = hf;
  m_hf2d_h    = hf_ptr;

  gpuQ( hipMalloc( (void**)&m_hf2d_d, sizeof( FH2D ) ) );
  gpuQ( hipMemcpy( m_hf2d_d, m_hf2d_h, sizeof( FH2D ), hipMemcpyHostToDevice ) );
}

void LoadGpuFuncHist::LD() {
  // this call  assume  already have Histofuncs set in m_hf
  // this function allocate memory of GPU and deep copy m_hf to m_hf_d

  if ( !m_hf ) {
    std::cout << "Error Load WiggleHistoFunctions " << std::endl;
    return;
  }

  FHs* hf = new FHs;

  hf->s_MaxValue       = ( *m_hf ).s_MaxValue;
  hf->nhist            = ( *m_hf ).nhist;
  hf->mxsz             = ( *m_hf ).mxsz;
  unsigned int* h_szs = ( *m_hf ).h_szs; // already allocateded on host ;


  // for (int i=0; i<hf->nhist; ++i) {
  //   std::cout << " " << m_hf->low_edge[i] << "\n";
  // }
  
  gpuQ( hipMalloc( (void**)&hf->low_edge, ( hf->nhist + 1 ) * sizeof( float ) ) );
  gpuQ( hipMemcpy( hf->low_edge, ( *m_hf ).low_edge, ( hf->nhist + 1 ) * sizeof( float ), hipMemcpyHostToDevice ) );

  gpuQ( hipMalloc( (void**)&hf->h_szs, hf->nhist * sizeof( unsigned int ) ) );
  gpuQ( hipMemcpy( hf->h_szs, ( *m_hf ).h_szs, hf->nhist * sizeof( unsigned int ), hipMemcpyHostToDevice ) );
  
  gpuQ( hipMalloc( &hf->d_contents1D, hf->nhist * hf->mxsz * sizeof( uint32_t ) ) );
  gpuQ( hipMalloc( &hf->d_borders1D, hf->nhist * hf->mxsz * sizeof( float ) ) );

  for ( size_t i = 0; i < hf->nhist; ++i ) {
    gpuQ( hipMemcpy( &( hf->d_contents1D[i * hf->mxsz] ), ( *m_hf ).h_contents[i], h_szs[i] * sizeof( uint32_t ),
                      hipMemcpyHostToDevice ) );
    gpuQ( hipMemcpy( &( hf->d_borders1D[i * hf->mxsz] ), ( *m_hf ).h_borders[i], h_szs[i] * sizeof( float ),
                      hipMemcpyHostToDevice ) );
  }

  // uint32_t tmp[hf->nhist*hf->mxsz];
  // float tmp[hf->nhist*hf->mxsz];
  // gpuQ( hipMemcpy(&tmp, (hf->d_borders1D), hf->nhist*hf->mxsz*sizeof(float), hipMemcpyDeviceToHost) );
  // for (int i=0; i<hf->nhist*hf->mxsz; ++i) {
  //   std::cout << " " << i << " " << tmp[i] << "\n";
  // }
  

  m_hf_h      = hf;

  gpuQ( hipMalloc( (void**)&m_hf_d, sizeof( FHs ) ) );
  gpuQ( hipMemcpy( m_hf_d, m_hf_h, sizeof( FHs ), hipMemcpyHostToDevice ) );

  // std::cout << "================== LoadGpuFuncHist::LD =======================\n";
  // std::cout << m_hf_h->nhist << "\n";
  // for (int i=0; i<m_hf_h->nhist; ++i) {
  //   std::cout << "h_szs: " << m_hf_h->h_szs[i] << "\n";
  //   for (int j=0; j<m_hf_h->mxsz; ++j) {
  //     std::cout << "  " << m_hf_h->h_contents[i][j] << "\n";
  //   }
  // }

}
