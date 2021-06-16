#include "LoadGpuFuncHist.h"
#include "gpuQ.h"
#include "hip/hip_runtime.h"
#include <iostream>

LoadGpuFuncHist::~LoadGpuFuncHist() {
  free( m_hf );
  hipFree( ( *m_hf_d ).low_edge );
  hipFree( ( *m_hf_d ).h_szs );
  for ( unsigned int i = 0; i < ( *m_d_hf ).nhist; ++i ) {
    hipFree( ( *m_hf_d ).h_contents[i] );
    hipFree( ( *m_hf_d ).h_borders[i] );
  }
  free( m_hf_d );
  hipFree( m_d_hf );

  free( m_hf2d );
  hipFree( ( *m_hf2d_d ).h_bordersx );
  hipFree( ( *m_hf2d_d ).h_bordersy );
  hipFree( ( *m_hf2d_d ).h_contents );
  free( m_hf2d_d );
  hipFree( m_d_hf2d );
}

void LoadGpuFuncHist::LD2D() {
  if ( !m_hf2d ) {
    std::cout << "Error Load 2DFunctionHisto " << std::endl;
    return;
  }

  FH2D* hf_ptr = new FH2D;
  FH2D  hf     = {0, 0, 0, 0, 0};

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
  m_hf2d_d    = hf_ptr;

  gpuQ( hipMalloc( (void**)&m_d_hf2d, sizeof( FH2D ) ) );
  gpuQ( hipMemcpy( m_d_hf2d, m_hf2d_d, sizeof( FH2D ), hipMemcpyHostToDevice ) );
}

void LoadGpuFuncHist::LD() {
  // this call  assume  already have Histofuncs set in m_hf
  // this function allocate memory of GPU and deep copy m_hf to m_d_hf

  if ( !m_hf ) {
    std::cout << "Error Load WiggleHistoFunctions " << std::endl;
    return;
  }

  FHs hf              = {0, 0, 0, 0, 0, 0};
  hf.s_MaxValue       = ( *m_hf ).s_MaxValue;
  hf.nhist            = ( *m_hf ).nhist;
  unsigned int* h_szs = ( *m_hf ).h_szs; // already allocateded on host ;

  gpuQ( hipMalloc( (void**)&hf.low_edge, ( hf.nhist + 1 ) * sizeof( float ) ) );
  gpuQ( hipMemcpy( hf.low_edge, ( *m_hf ).low_edge, ( hf.nhist + 1 ) * sizeof( float ), hipMemcpyHostToDevice ) );

  gpuQ( hipMalloc( (void**)&hf.h_szs, hf.nhist * sizeof( unsigned int ) ) );
  gpuQ( hipMemcpy( hf.h_szs, ( *m_hf ).h_szs, hf.nhist * sizeof( unsigned int ), hipMemcpyHostToDevice ) );

  gpuQ( hipMalloc( (void**)&hf.h_contents, hf.nhist * sizeof( uint32_t* ) ) );
  gpuQ( hipMalloc( (void**)&hf.h_borders, hf.nhist * sizeof( float* ) ) );

  uint32_t** contents_ptr = (uint32_t**)malloc( hf.nhist * sizeof( uint32_t* ) );
  float**    borders_ptr  = (float**)malloc( hf.nhist * sizeof( float* ) );

  for ( unsigned int i = 0; i < hf.nhist; ++i ) {

    gpuQ( hipMalloc( (void**)( contents_ptr + i ), h_szs[i] * sizeof( uint32_t ) ) );
    gpuQ( hipMalloc( (void**)&( borders_ptr[i] ), ( h_szs[i] + 1 ) * sizeof( float ) ) );
    gpuQ(
        hipMemcpy( contents_ptr[i], ( *m_hf ).h_contents[i], h_szs[i] * sizeof( uint32_t ), hipMemcpyHostToDevice ) );
    gpuQ( hipMemcpy( borders_ptr[i], ( *m_hf ).h_borders[i], ( h_szs[i] + 1 ) * sizeof( float ),
                      hipMemcpyHostToDevice ) );
    // std::cout << ".....Loading  WiggleFunctionHistss, Size of Hists[" << i << "]=" << h_szs[i]<< std::endl ;
  }

  gpuQ( hipMemcpy( hf.h_contents, contents_ptr, hf.nhist * sizeof( uint32_t* ), hipMemcpyHostToDevice ) );
  gpuQ( hipMemcpy( hf.h_borders, borders_ptr, hf.nhist * sizeof( float* ), hipMemcpyHostToDevice ) );

  gpuQ( hipMalloc( (void**)&m_d_hf, sizeof( FHs ) ) );
  gpuQ( hipMemcpy( m_d_hf, &hf, sizeof( FHs ), hipMemcpyHostToDevice ) );

  free( contents_ptr );
  free( borders_ptr );

  m_hf_d = &hf;
}
