/*
  Copyright (C) 2002-2021 CERN for the benefit of the ATLAS collaboration
*/

#include "LoadGpuFuncHist.h"

#include <omp.h>
#include <iostream>

LoadGpuFuncHist::LoadGpuFuncHist() {
#ifdef USE_OMPGPU
    select_omp_device();
#endif
}

  LoadGpuFuncHist::~LoadGpuFuncHist() {
  free( m_hf );
  omp_target_free ( ( *m_hf_h ).low_edge,     m_select_device ); 
  omp_target_free ( ( *m_hf_h ).h_szs,        m_select_device ); 
  omp_target_free ( ( *m_hf_h ).h_contents,   m_select_device ); 
  omp_target_free ( ( *m_hf_h ).h_borders,    m_select_device ); 
  omp_target_free ( ( *m_hf_h ).d_contents1D, m_select_device ); 
  omp_target_free ( ( *m_hf_h ).d_borders1D,  m_select_device ); 
  free( m_hf_h );
  omp_target_free ( m_hf_d,  m_select_device ); 

  free( m_hf2d );
  omp_target_free ( ( *m_hf2d_h ).h_bordersx,  m_select_device ); 
  omp_target_free ( ( *m_hf2d_h ).h_bordersy,  m_select_device ); 
  omp_target_free ( ( *m_hf2d_h ).h_contents,  m_select_device ); 
  free( m_hf2d_h );
  omp_target_free ( m_hf2d_d,  m_select_device ); 
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

    hf.h_bordersx = (float *) omp_target_alloc( ( hf.nbinsx + 1 ) * sizeof( float ), m_select_device );
    hf.h_bordersy = (float *) omp_target_alloc( ( hf.nbinsy + 1 ) * sizeof( float ), m_select_device );
    hf.h_contents = (float *) omp_target_alloc( ( hf.nbinsy * hf.nbinsx ) * sizeof( float ), m_select_device );
    if ( hf.h_bordersx == NULL or hf.h_bordersy == NULL or hf.h_contents == NULL ) {
      std::cout << " ERROR: No space left on device." << std::endl;;
      //return false;
    }
   
//  //  std::cout << "hf.h_bordersy, pointer " <<  hf.h_bordersy  <<  std::endl ;
    if ( omp_target_memcpy( hf.h_bordersx, ( *m_hf2d ).h_bordersx, ( hf.nbinsx + 1 ) * sizeof( float ),
                                            m_offset, m_offset, m_select_device, m_initial_device ) ) {
      std::cout << "ERROR: copy hf.nbinsx. " << std::endl;
      //std::exit(1);
    }
    if ( omp_target_memcpy( hf.h_bordersy, ( *m_hf2d ).h_bordersy, ( hf.nbinsy + 1 ) * sizeof( float ),
                                            m_offset, m_offset, m_select_device, m_initial_device ) ) {
      std::cout << "ERROR: copy hf.nbinsy. " << std::endl;
      //std::exit(1);
    }
    if ( omp_target_memcpy( hf.h_contents, ( *m_hf2d ).h_contents, ( hf.nbinsx * hf.nbinsy ) * sizeof( float ),
                                            m_offset, m_offset, m_select_device, m_initial_device ) ) {
      std::cout << "ERROR: copy hf.h_contents. " << std::endl;
      //std::exit(1);
    }

    *( hf_ptr ) = hf;
    m_hf2d_h    = hf_ptr;

    m_hf2d_d = (FH2D *) omp_target_alloc( sizeof( FH2D ), m_select_device );
    if ( m_hf2d_d == NULL ) {
      std::cout << " ERROR: No space left on device." << std::endl;;
    }
    if ( omp_target_memcpy( m_hf2d_d, m_hf2d_h, sizeof( FH2D ), m_offset, m_offset, m_select_device, m_initial_device ) ) {
      std::cout << "ERROR: copy m_hf2d_d. " << std::endl;
      //std::exit(1);
    }
}

void LoadGpuFuncHist::LD() {
  // this call  assume  already have Histofuncs set in m_hf
  // this function allocate memory of GPU and deep copy m_hf to m_hf_d

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

    hf.low_edge = (float *) omp_target_alloc( ( hf.nhist + 1 ) * sizeof( float ), m_select_device );
    if ( hf.low_edge == NULL ) {
      std::cout << " ERROR: No space left on device." << std::endl;;
      //return false;
    }
    if ( omp_target_memcpy( hf.low_edge, ( *m_hf ).low_edge, ( hf.nhist + 1 ) * sizeof( float ),
                                            m_offset, m_offset, m_select_device, m_initial_device ) ) {
      std::cout << "ERROR: copy hf.low_edge. " << std::endl;
      //std::exit(1);
    }

    hf.h_szs = (unsigned int *) omp_target_alloc(  hf.nhist * sizeof( unsigned int ), m_select_device );
    if ( hf.h_szs == NULL ) {
      std::cout << " ERROR: No space left on device." << std::endl;;
      //return false;
    }
    if ( omp_target_memcpy( hf.h_szs, ( *m_hf ).h_szs, hf.nhist * sizeof( unsigned int ),
                                     m_offset, m_offset, m_select_device, m_initial_device ) ) {
      std::cout << "ERROR: copy hf.h_szs. " << std::endl;
      //std::exit(1);
    }

    hf.d_contents1D = (uint32_t *) omp_target_alloc( hf.nhist * hf.mxsz * sizeof( uint32_t ), m_select_device );
    hf.d_borders1D  =    (float *) omp_target_alloc( hf.nhist * hf.mxsz * sizeof( float ), m_select_device );
    if ( hf.d_contents1D == NULL or hf.d_contents1D == NULL ) {
      std::cout << " ERROR: No space left on device." << std::endl;;
      //return false;
    }

  for ( size_t i = 0; i < hf.nhist; ++i ) {
      if ( omp_target_memcpy( &( hf.d_contents1D[i * hf.mxsz] ), ( *m_hf ).h_contents[i], h_szs[i] * sizeof( uint32_t ),
                              m_offset, m_offset, m_select_device, m_initial_device ) ) {
      std::cout << "ERROR: copy hf.d_contents1D. " << std::endl;
      //std::exit(1);
      }
      if ( omp_target_memcpy( &( hf.d_borders1D[i * hf.mxsz] ), ( *m_hf ).h_borders[i], h_szs[i] * sizeof( float ),
                              m_offset, m_offset, m_select_device, m_initial_device ) ) {
      std::cout << "ERROR: copy hf.d_borders1D " << std::endl;
      //std::exit(1);
      }
  }

  *( hf_ptr ) = hf;
  m_hf_h      = hf_ptr;

    m_hf_d = (FHs *) omp_target_alloc( sizeof( FHs ), m_select_device );
    if ( m_hf_d == NULL ) {
      std::cout << " ERROR: No space left on device." << std::endl;;
      //return false;
    }
    if ( omp_target_memcpy( m_hf_d, m_hf_h, sizeof( FHs ), m_offset, m_offset, m_select_device, m_initial_device ) ) {
      std::cout << "ERROR: copy hf.h_szs. " << std::endl;
      //std::exit(1);
    }
}

