#include "LoadGpuFuncHist.h"

LoadGpuFuncHist::~LoadGpuFuncHist() {
  free( m_hf );
  cudaFree( ( *m_hf_d ).low_edge );
  cudaFree( ( *m_hf_d ).h_szs );
  for ( unsigned int i = 0; i < ( *m_d_hf ).nhist; ++i ) {
    cudaFree( ( *m_hf_d ).h_contents[i] );
    cudaFree( ( *m_hf_d ).h_borders[i] );
  }
  free( m_hf_d );
  cudaFree( m_d_hf );

  free( m_hf2d );
  cudaFree( ( *m_hf2d_d ).h_bordersx );
  cudaFree( ( *m_hf2d_d ).h_bordersy );
  cudaFree( ( *m_hf2d_d ).h_contents );
  free( m_hf2d_d );
  cudaFree( m_d_hf2d );
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

  gpuQ( cudaMalloc( (void**)&hf.h_bordersx, ( hf.nbinsx + 1 ) * sizeof( float ) ) );
  gpuQ( cudaMalloc( (void**)&hf.h_bordersy, ( hf.nbinsy + 1 ) * sizeof( float ) ) );
  gpuQ( cudaMalloc( (void**)&hf.h_contents, ( hf.nbinsy * hf.nbinsx ) * sizeof( float ) ) );
  gpuQ( cudaMemcpy( hf.h_bordersx, ( *m_hf2d ).h_bordersx, ( hf.nbinsx + 1 ) * sizeof( float ),
                    cudaMemcpyHostToDevice ) );
  //  std::cout << "hf.h_bordersy, pointer " <<  hf.h_bordersy  <<  std::endl ;
  gpuQ( cudaMemcpy( hf.h_bordersy, ( *m_hf2d ).h_bordersy, ( hf.nbinsy + 1 ) * sizeof( float ),
                    cudaMemcpyHostToDevice ) );
  gpuQ( cudaMemcpy( hf.h_contents, ( *m_hf2d ).h_contents, ( hf.nbinsx * hf.nbinsy ) * sizeof( float ),
                    cudaMemcpyHostToDevice ) );
  *( hf_ptr ) = hf;
  m_hf2d_d    = hf_ptr;

  gpuQ( cudaMalloc( (void**)&m_d_hf2d, sizeof( FH2D ) ) );
  gpuQ( cudaMemcpy( m_d_hf2d, m_hf2d_d, sizeof( FH2D ), cudaMemcpyHostToDevice ) );
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

  gpuQ( cudaMalloc( (void**)&hf.low_edge, ( hf.nhist + 1 ) * sizeof( float ) ) );
  gpuQ( cudaMemcpy( hf.low_edge, ( *m_hf ).low_edge, ( hf.nhist + 1 ) * sizeof( float ), cudaMemcpyHostToDevice ) );

  gpuQ( cudaMalloc( (void**)&hf.h_szs, hf.nhist * sizeof( unsigned int ) ) );
  gpuQ( cudaMemcpy( hf.h_szs, ( *m_hf ).h_szs, hf.nhist * sizeof( unsigned int ), cudaMemcpyHostToDevice ) );

  gpuQ( cudaMalloc( (void**)&hf.h_contents, hf.nhist * sizeof( uint32_t* ) ) );
  gpuQ( cudaMalloc( (void**)&hf.h_borders, hf.nhist * sizeof( float* ) ) );

  uint32_t** contents_ptr = (uint32_t**)malloc( hf.nhist * sizeof( uint32_t* ) );
  float**    borders_ptr  = (float**)malloc( hf.nhist * sizeof( float* ) );

  for ( unsigned int i = 0; i < hf.nhist; ++i ) {

    gpuQ( cudaMalloc( (void**)( contents_ptr + i ), h_szs[i] * sizeof( uint32_t ) ) );
    gpuQ( cudaMalloc( (void**)&( borders_ptr[i] ), ( h_szs[i] + 1 ) * sizeof( float ) ) );
    gpuQ(
        cudaMemcpy( contents_ptr[i], ( *m_hf ).h_contents[i], h_szs[i] * sizeof( uint32_t ), cudaMemcpyHostToDevice ) );
    gpuQ( cudaMemcpy( borders_ptr[i], ( *m_hf ).h_borders[i], ( h_szs[i] + 1 ) * sizeof( float ),
                      cudaMemcpyHostToDevice ) );
    // std::cout << ".....Loading  WiggleFunctionHistss, Size of Hists[" << i << "]=" << h_szs[i]<< std::endl ;
  }

  gpuQ( cudaMemcpy( hf.h_contents, contents_ptr, hf.nhist * sizeof( uint32_t* ), cudaMemcpyHostToDevice ) );
  gpuQ( cudaMemcpy( hf.h_borders, borders_ptr, hf.nhist * sizeof( float* ), cudaMemcpyHostToDevice ) );

  gpuQ( cudaMalloc( (void**)&m_d_hf, sizeof( FHs ) ) );
  gpuQ( cudaMemcpy( m_d_hf, &hf, sizeof( FHs ), cudaMemcpyHostToDevice ) );

  free( contents_ptr );
  free( borders_ptr );

  m_hf_d = &hf;
}
