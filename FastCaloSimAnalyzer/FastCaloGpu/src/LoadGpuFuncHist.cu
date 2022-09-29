/*
  Copyright (C) 2002-2021 CERN for the benefit of the ATLAS collaboration
*/

#include "LoadGpuFuncHist.h"
#include <iostream>
#include "gpuQ.h"
#include "DEV_BigMem.h"

DEV_BigMem* DEV_BigMem::bm_ptr;

LoadGpuFuncHist::LoadGpuFuncHist() {}

LoadGpuFuncHist::~LoadGpuFuncHist() {
  free( m_hf );
  cudaFree( m_hf_d );

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
  FH2D  hf     = {0, 0, 0, 0, 0};

  hf.nbinsx = ( *m_hf2d ).nbinsx;
  hf.nbinsy = ( *m_hf2d ).nbinsy;

  DEV_BigMem* p = DEV_BigMem::bm_ptr;

  hf.h_bordersx = (float*)( p->dev_bm_alloc( ( hf.nbinsx + 1 ) * sizeof( float ) ) );
  hf.h_bordersy = (float*)( p->dev_bm_alloc( ( hf.nbinsy + 1 ) * sizeof( float ) ) );
  hf.h_contents = (float*)( p->dev_bm_alloc( hf.nbinsy * hf.nbinsx * sizeof( float ) ) );

  gpuQ( cudaMemcpy( hf.h_bordersx, ( *m_hf2d ).h_bordersx, ( hf.nbinsx + 1 ) * sizeof( float ),
                    cudaMemcpyHostToDevice ) );
  gpuQ( cudaMemcpy( hf.h_bordersy, ( *m_hf2d ).h_bordersy, ( hf.nbinsy + 1 ) * sizeof( float ),
                    cudaMemcpyHostToDevice ) );
  gpuQ( cudaMemcpy( hf.h_contents, ( *m_hf2d ).h_contents, ( hf.nbinsx * hf.nbinsy ) * sizeof( float ),
                    cudaMemcpyHostToDevice ) );

  *( hf_ptr ) = hf;
  m_hf2d_d    = hf_ptr;

  m_d_hf2d = (FH2D*)( p->dev_bm_alloc( sizeof( FH2D ) ) );
  gpuQ( cudaMemcpy( m_d_hf2d, m_hf2d_d, sizeof( FH2D ), cudaMemcpyHostToDevice ) );
}

/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

void LoadGpuFuncHist::LD() {
  // this call  assume  already have Histofuncs set in m_hf
  // this function allocate memory of GPU and deep copy m_hf to m_hf_d

  if ( !m_hf ) {
    std::cout << "Error Load WiggleHistoFunctions " << std::endl;
    return;
  }

  FHs hf              = {0, 0, 0, 0, 0, 0};
  hf.s_MaxValue       = ( *m_hf ).s_MaxValue;
  hf.nhist            = ( *m_hf ).nhist;
  unsigned int* h_szs = ( *m_hf ).h_szs; // already allocateded on host ;

  DEV_BigMem* p = DEV_BigMem::bm_ptr;

  hf.low_edge = (float*)( p->dev_bm_alloc( ( hf.nhist + 1 ) * sizeof( float ) ) );
  gpuQ( cudaMemcpy( hf.low_edge, ( *m_hf ).low_edge, ( hf.nhist + 1 ) * sizeof( float ), cudaMemcpyHostToDevice ) );

  hf.h_szs = (unsigned int*)( p->dev_bm_alloc( hf.nhist * sizeof( float ) ) );
  gpuQ( cudaMemcpy( hf.h_szs, ( *m_hf ).h_szs, hf.nhist * sizeof( unsigned int ), cudaMemcpyHostToDevice ) );

  hf.h_contents = (uint32_t**)( p->dev_bm_alloc( hf.nhist * sizeof( uint32_t* ) ) );
  hf.h_borders  = (float**)( p->dev_bm_alloc( hf.nhist * sizeof( float* ) ) );

  uint32_t** contents_ptr = (uint32_t**)malloc( hf.nhist * sizeof( uint32_t* ) );
  float**    borders_ptr  = (float**)malloc( hf.nhist * sizeof( float* ) );

  for ( unsigned int i = 0; i < hf.nhist; ++i ) {

    contents_ptr[i] = (uint32_t*)( p->dev_bm_alloc( h_szs[i] * sizeof( uint32_t ) ) );
    borders_ptr[i]  = (float*)( p->dev_bm_alloc( ( h_szs[i] + 1 ) * sizeof( float ) ) );

    gpuQ(
        cudaMemcpy( contents_ptr[i], ( *m_hf ).h_contents[i], h_szs[i] * sizeof( uint32_t ), cudaMemcpyHostToDevice ) );
    gpuQ( cudaMemcpy( borders_ptr[i], ( *m_hf ).h_borders[i], ( h_szs[i] + 1 ) * sizeof( float ),
                      cudaMemcpyHostToDevice ) );

    // for (int j=0; j<h_szs[i]; ++j) {
    //   std::cout << i << " " << j << " " << m_hf->h_contents[i][j] << " " << m_hf->h_borders[i][j]
    //             << " hbrds\n";
    // }
    
  }

  gpuQ( cudaMemcpy( hf.h_contents, contents_ptr, hf.nhist * sizeof( uint32_t* ), cudaMemcpyHostToDevice ) );
  gpuQ( cudaMemcpy( hf.h_borders, borders_ptr, hf.nhist * sizeof( float* ), cudaMemcpyHostToDevice ) );

  m_d_hf = (FHs*)( p->dev_bm_alloc( sizeof( FHs ) ) );
  gpuQ( cudaMemcpy( m_d_hf, &hf, sizeof( FHs ), cudaMemcpyHostToDevice ) );

  free( contents_ptr );
  free( borders_ptr );

  m_hf_d = &hf;

  // std::cout << "LD1D: nhist: "<<hf.nhist<<"   memeory: " <<s << " M of FHs str: "<< sizeof(FHs)  <<std::endl ;

  // std::cout << "================== LoadGpuFuncHist::LD =======================\n";
  // std::cout << m_hf_h->nhist << "\n";
  // for (int i=0; i<m_hf_h->nhist; ++i) {
  //   std::cout << "h_szs: " << m_hf_h->h_szs[i] << "\n";
  //   for (int j=0; j<m_hf_h->mxsz; ++j) {
  //     std::cout << "  " << m_hf_h->h_contents[i][j] << "\n";
  //   }
  // }
}
