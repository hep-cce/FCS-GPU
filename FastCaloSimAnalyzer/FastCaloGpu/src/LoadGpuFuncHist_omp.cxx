/*
  Copyright (C) 2002-2021 CERN for the benefit of the ATLAS collaboration
*/

#include "LoadGpuFuncHist.h"

#include <omp.h>
#include <iostream>

LoadGpuFuncHist::LoadGpuFuncHist() {}

  LoadGpuFuncHist::~LoadGpuFuncHist() {
//  free( m_hf );
//  cudaFree( ( *m_hf_h ).low_edge );
//  cudaFree( ( *m_hf_h ).h_szs );
//  cudaFree( ( *m_hf_h ).h_contents );
//  cudaFree( ( *m_hf_h ).h_borders );
//  cudaFree( ( *m_hf_h ).d_contents1D );
//  cudaFree( ( *m_hf_h ).d_borders1D );
//  free( m_hf_h );
//  cudaFree( m_hf_d );
//
//  free( m_hf2d );
//  cudaFree( ( *m_hf2d_h ).h_bordersx );
//  cudaFree( ( *m_hf2d_h ).h_bordersy );
//  cudaFree( ( *m_hf2d_h ).h_contents );
//  free( m_hf2d_h );
//  cudaFree( m_hf2d_d );
}

void LoadGpuFuncHist::LD2D() {
//  if ( !m_hf2d ) {
//    std::cout << "Error Load 2DFunctionHisto " << std::endl;
//    return;
//  }
//
//  FH2D* hf_ptr = new FH2D;
//  FH2D  hf;
//
//  hf.nbinsx = ( *m_hf2d ).nbinsx;
//  hf.nbinsy = ( *m_hf2d ).nbinsy;
//  // std::cout << ".....Loading  2DFnctionHists, Size of hist" <<  hf.nbinsx  << "x" << hf.nbinsy << std::endl ;
//  // std::cout << "(*m_hf2d).h_bordersy, pointer " <<  (*m_hf2d).h_bordersy   << std::endl ;
//
//  gpuQ( cudaMalloc( (void**)&hf.h_bordersx, ( hf.nbinsx + 1 ) * sizeof( float ) ) );
//  gpuQ( cudaMalloc( (void**)&hf.h_bordersy, ( hf.nbinsy + 1 ) * sizeof( float ) ) );
//  gpuQ( cudaMalloc( (void**)&hf.h_contents, ( hf.nbinsy * hf.nbinsx ) * sizeof( float ) ) );
//  gpuQ( cudaMemcpy( hf.h_bordersx, ( *m_hf2d ).h_bordersx, ( hf.nbinsx + 1 ) * sizeof( float ),
//                    cudaMemcpyHostToDevice ) );
//  //  std::cout << "hf.h_bordersy, pointer " <<  hf.h_bordersy  <<  std::endl ;
//  gpuQ( cudaMemcpy( hf.h_bordersy, ( *m_hf2d ).h_bordersy, ( hf.nbinsy + 1 ) * sizeof( float ),
//                    cudaMemcpyHostToDevice ) );
//  gpuQ( cudaMemcpy( hf.h_contents, ( *m_hf2d ).h_contents, ( hf.nbinsx * hf.nbinsy ) * sizeof( float ),
//                    cudaMemcpyHostToDevice ) );
//  *( hf_ptr ) = hf;
//  m_hf2d_h    = hf_ptr;
//
//  gpuQ( cudaMalloc( (void**)&m_hf2d_d, sizeof( FH2D ) ) );
//  gpuQ( cudaMemcpy( m_hf2d_d, m_hf2d_h, sizeof( FH2D ), cudaMemcpyHostToDevice ) );
}

void LoadGpuFuncHist::LD() {
  // this call  assume  already have Histofuncs set in m_hf
  // this function allocate memory of GPU and deep copy m_hf to m_hf_d

  //TODO: Wrap device IDs from omp APIs to a class
  //link to GeoLoadGpu.h, Rand4Hits.h, LoadGpuFuncHist
  int num_devices    = omp_get_num_devices();
  int initial_device = omp_get_initial_device();
  int default_device = omp_get_default_device();
  std::size_t offset = 0;

  if ( !m_hf ) {
    std::cout << "Error Load WiggleHistoFunctions " << std::endl;
    return;
  }

  FHs* hf_ptr = new FHs;
  FHs hf;

  hf.s_MaxValue       = ( *m_hf ).s_MaxValue;
  hf.nhist            = ( *m_hf ).nhist;
  hf.mxsz             = ( *m_hf ).mxsz;
  unsigned int* h_szs = ( *m_hf ).h_szs; // already allocateded on host ;

//  gpuQ( cudaMalloc( (void**)&hf.low_edge, ( hf.nhist + 1 ) * sizeof( float ) ) );
//  gpuQ( cudaMemcpy( hf.low_edge, ( *m_hf ).low_edge, ( hf.nhist + 1 ) * sizeof( float ), cudaMemcpyHostToDevice ) );
    hf.low_edge = (float *) omp_target_alloc( ( hf.nhist + 1 ) * sizeof( float ), default_device );
    if ( hf.low_edge == NULL ) {
      std::cout << " ERROR: No space left on device." << std::endl;;
      //return false;
    }
    if ( omp_target_memcpy( hf.low_edge, ( *m_hf ).low_edge, ( hf.nhist + 1 ) * sizeof( float ),
                                            offset, offset, default_device, initial_device ) ) {
      std::cout << "ERROR: copy hf.low_edge. " << std::endl;
      //std::exit(1);
    }

//  gpuQ( cudaMalloc( (void**)&hf.h_szs, hf.nhist * sizeof( unsigned int ) ) );
//  gpuQ( cudaMemcpy( hf.h_szs, ( *m_hf ).h_szs, hf.nhist * sizeof( unsigned int ), cudaMemcpyHostToDevice ) );
    hf.h_szs = (unsigned int *) omp_target_alloc(  hf.nhist * sizeof( unsigned int ), default_device );
    if ( hf.h_szs == NULL ) {
      std::cout << " ERROR: No space left on device." << std::endl;;
      //return false;
    }
    if ( omp_target_memcpy( hf.h_szs, ( *m_hf ).h_szs, hf.nhist * sizeof( unsigned int ),
                                     offset, offset, default_device, initial_device ) ) {
      std::cout << "ERROR: copy hf.h_szs. " << std::endl;
      //std::exit(1);
    }

//  gpuQ( cudaMalloc( &hf.d_contents1D, hf.nhist * hf.mxsz * sizeof( uint32_t ) ) );
//  gpuQ( cudaMalloc( &hf.d_borders1D, hf.nhist * hf.mxsz * sizeof( float ) ) );
    hf.d_contents1D = (uint32_t *) omp_target_alloc( hf.nhist * hf.mxsz * sizeof( uint32_t ), default_device );
    hf.d_borders1D  =    (float *) omp_target_alloc( hf.nhist * hf.mxsz * sizeof( float ), default_device );
    if ( hf.d_contents1D == NULL or hf.d_contents1D == NULL ) {
      std::cout << " ERROR: No space left on device." << std::endl;;
      //return false;
    }

  for ( size_t i = 0; i < hf.nhist; ++i ) {
//    gpuQ( cudaMemcpy( &( hf.d_contents1D[i * hf.mxsz] ), ( *m_hf ).h_contents[i], h_szs[i] * sizeof( uint32_t ),
//                      cudaMemcpyHostToDevice ) );
//    gpuQ( cudaMemcpy( &( hf.d_borders1D[i * hf.mxsz] ), ( *m_hf ).h_borders[i], h_szs[i] * sizeof( float ),
//                      cudaMemcpyHostToDevice ) );
      if ( omp_target_memcpy( &( hf.d_contents1D[i * hf.mxsz] ), ( *m_hf ).h_contents[i], h_szs[i] * sizeof( uint32_t ),
                              offset, offset, default_device, initial_device ) ) {
      std::cout << "ERROR: copy hf.d_contents1D. " << std::endl;
      //std::exit(1);
      }
      if ( omp_target_memcpy( &( hf.d_borders1D[i * hf.mxsz] ), ( *m_hf ).h_borders[i], h_szs[i] * sizeof( float ),
                              offset, offset, default_device, initial_device ) ) {
      std::cout << "ERROR: copy hf.d_borders1D " << std::endl;
      //std::exit(1);
      }
  }

  *( hf_ptr ) = hf;
  m_hf_h      = hf_ptr;

//  gpuQ( cudaMalloc( (void**)&m_hf_d, sizeof( FHs ) ) );
//  gpuQ( cudaMemcpy( m_hf_d, m_hf_h, sizeof( FHs ), cudaMemcpyHostToDevice ) );
    m_hf_d = (FHs *) omp_target_alloc( sizeof( FHs ), default_device );
    if ( m_hf_d == NULL ) {
      std::cout << " ERROR: No space left on device." << std::endl;;
      //return false;
    }
    if ( omp_target_memcpy( m_hf_d, m_hf_h, sizeof( FHs ), offset, offset, default_device, initial_device ) ) {
      std::cout << "ERROR: copy hf.h_szs. " << std::endl;
      //std::exit(1);
    }
}

