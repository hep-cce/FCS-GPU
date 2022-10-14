/*
  Copyright (C) 2002-2022 CERN for the benefit of the ATLAS collaboration
*/

#include "LoadGpuFuncHist.h"
#include "AlpakaDefs.h"
#include <alpaka/alpaka.hpp>
#include <iostream>

class LoadGpuFuncHist::Impl {
public:
  Impl()
    : bufBordersX_acc{alpaka::allocBuf<float, Idx>(alpaka::getDevByIdx<Acc>(0u), Idx{1})}
    , bufBordersY_acc{alpaka::allocBuf<float, Idx>(alpaka::getDevByIdx<Acc>(0u), Idx{1})}
    , bufContents_acc{alpaka::allocBuf<float, Idx>(alpaka::getDevByIdx<Acc>(0u), Idx{1})}
    , bufFH2D_acc{alpaka::allocBuf<FH2D, Idx>(alpaka::getDevByIdx<Acc>(0u), Idx{1})}
    , bufLowEdge_acc{alpaka::allocBuf<float, Idx>(alpaka::getDevByIdx<Acc>(0u), Idx{1})}
    , bufSzs_acc{alpaka::allocBuf<unsigned, Idx>(alpaka::getDevByIdx<Acc>(0u), Idx{1})}
    , bufContents1D_acc{alpaka::allocBuf<uint32_t, Idx>(alpaka::getDevByIdx<Acc>(0u), Idx{1})}
    , bufBorders1D_acc{alpaka::allocBuf<float, Idx>(alpaka::getDevByIdx<Acc>(0u), Idx{1})}
    , bufFHs_acc{alpaka::allocBuf<FHs, Idx>(alpaka::getDevByIdx<Acc>(0u), Idx{1})}
 {}

  // 2D
  BufAcc bufBordersX_acc;
  BufAcc bufBordersY_acc;
  BufAcc bufContents_acc;

  BufAccFH2D bufFH2D_acc;

  // 1D
  BufAcc bufLowEdge_acc;
  BufAccUnsigned bufSzs_acc;
  BufAccUint32 bufContents1D_acc;
  BufAcc bufBorders1D_acc;

  BufAccFHs bufFHs_acc;
};

LoadGpuFuncHist::LoadGpuFuncHist() {
  pImpl = new Impl();
}

LoadGpuFuncHist::~LoadGpuFuncHist() {
  free( m_hf );
  free( m_hf_h );
  free( m_hf2d );
  free( m_hf2d_h );
  delete pImpl;
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

  pImpl->bufBordersX_acc = alpaka::allocBuf<float, Idx>(alpaka::getDevByIdx<Acc>(0u), static_cast<Idx>( hf.nbinsx + 1 ));
  pImpl->bufBordersY_acc = alpaka::allocBuf<float, Idx>(alpaka::getDevByIdx<Acc>(0u), static_cast<Idx>( hf.nbinsy + 1 ));
  pImpl->bufContents_acc = alpaka::allocBuf<float, Idx>(alpaka::getDevByIdx<Acc>(0u), static_cast<Idx>( hf.nbinsy * hf.nbinsx ));
  hf.h_bordersx = alpaka::getPtrNative(pImpl->bufBordersX_acc);
  hf.h_bordersy = alpaka::getPtrNative(pImpl->bufBordersY_acc);
  hf.h_contents = alpaka::getPtrNative(pImpl->bufContents_acc);

  
  BufHost bufBordersX_host = alpaka::allocBuf<float, Idx>(alpaka::getDevByIdx<Host>(0u), static_cast<Idx>( hf.nbinsx + 1 ));
  float* bufBordersX_host_ptr = alpaka::getPtrNative(bufBordersX_host);
  std::memcpy(bufBordersX_host_ptr,( *m_hf2d ).h_bordersx, ( hf.nbinsx + 1 ) * sizeof( float ));

  BufHost bufBordersY_host = alpaka::allocBuf<float, Idx>(alpaka::getDevByIdx<Host>(0u), static_cast<Idx>( hf.nbinsy + 1 ));
  float* bufBordersY_host_ptr = alpaka::getPtrNative(bufBordersY_host);
  std::memcpy(bufBordersY_host_ptr,( *m_hf2d ).h_bordersy, ( hf.nbinsy + 1 ) * sizeof( float ));

  BufHost bufContents_host = alpaka::allocBuf<float, Idx>(alpaka::getDevByIdx<Host>(0u), static_cast<Idx>( hf.nbinsx * hf.nbinsy ));
  float* bufContents_host_ptr = alpaka::getPtrNative(bufContents_host);
  std::memcpy(bufContents_host_ptr,( *m_hf2d ).h_contents, ( hf.nbinsx * hf.nbinsy ) * sizeof( float ));

  QueueAcc queue(alpaka::getDevByIdx<Acc>(Idx{0}));
  alpaka::memcpy(queue,pImpl->bufBordersX_acc,bufBordersX_host);
  alpaka::memcpy(queue,pImpl->bufBordersY_acc,bufBordersY_host);
  alpaka::memcpy(queue,pImpl->bufContents_acc,bufContents_host);

  *( hf_ptr ) = hf;
  m_hf2d_h    = hf_ptr;

  m_hf2d_d = alpaka::getPtrNative(pImpl->bufFH2D_acc);
  BufHostFH2D bufFH2D_host = alpaka::allocBuf<FH2D, Idx>(alpaka::getDevByIdx<Host>(0u), Idx{1});
  FH2D* bufFH2D_host_ptr = alpaka::getPtrNative(bufFH2D_host);
  *bufFH2D_host_ptr = *m_hf2d_h;

  alpaka::memcpy(queue,pImpl->bufFH2D_acc,bufFH2D_host);
  alpaka::wait(queue);
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

  pImpl->bufLowEdge_acc = alpaka::allocBuf<float, Idx>(alpaka::getDevByIdx<Acc>(0u), static_cast<Idx>( hf->nhist + 1 ));
  pImpl->bufSzs_acc = alpaka::allocBuf<unsigned, Idx>(alpaka::getDevByIdx<Acc>(0u), static_cast<Idx>( hf->nhist ));
  pImpl->bufContents1D_acc = alpaka::allocBuf<uint32_t, Idx>(alpaka::getDevByIdx<Acc>(0u), static_cast<Idx>( hf->nhist * hf->mxsz ));
  pImpl->bufBorders1D_acc = alpaka::allocBuf<float, Idx>(alpaka::getDevByIdx<Acc>(0u), static_cast<Idx>( hf->nhist * hf->mxsz ));
  hf->low_edge= alpaka::getPtrNative(pImpl->bufLowEdge_acc);
  hf->h_szs = alpaka::getPtrNative(pImpl->bufSzs_acc);
  hf->d_contents1D = alpaka::getPtrNative(pImpl->bufContents1D_acc);
  hf->d_borders1D = alpaka::getPtrNative(pImpl->bufBorders1D_acc);


  BufHost bufLowEdge_host = alpaka::allocBuf<float, Idx>(alpaka::getDevByIdx<Host>(0u), static_cast<Idx>( hf->nhist + 1 ));
  float* bufLowEdge_host_ptr = alpaka::getPtrNative(bufLowEdge_host);
  std::memcpy(bufLowEdge_host_ptr,( *m_hf ).low_edge, ( hf->nhist + 1 ) * sizeof( float ));

  BufHostUnsigned bufSzs_host = alpaka::allocBuf<unsigned, Idx>(alpaka::getDevByIdx<Host>(0u), static_cast<Idx>( hf->nhist ));
  unsigned* bufSzs_host_ptr = alpaka::getPtrNative(bufSzs_host);
  std::memcpy(bufSzs_host_ptr, ( *m_hf ).h_szs, hf->nhist * sizeof( unsigned int ));

  QueueAcc queue(alpaka::getDevByIdx<Acc>(Idx{0}));
  alpaka::memcpy(queue,pImpl->bufLowEdge_acc,bufLowEdge_host);
  alpaka::memcpy(queue,pImpl->bufSzs_acc,bufSzs_host);

  BufHostUint32 bufContents1D_host = alpaka::allocBuf<uint32_t, Idx>(alpaka::getDevByIdx<Host>(0u), static_cast<Idx>( hf->nhist * hf->mxsz ));
  uint32_t* bufContents1D_host_ptr = alpaka::getPtrNative(bufContents1D_host);
  BufHost bufBorders1D_host = alpaka::allocBuf<float, Idx>(alpaka::getDevByIdx<Host>(0u), static_cast<Idx>( hf->nhist * hf->mxsz ));
  float* bufBorders1D_host_ptr = alpaka::getPtrNative(bufBorders1D_host);

  for ( size_t i = 0; i < hf->nhist; ++i ) {
    std::memcpy(&(bufContents1D_host_ptr[i * hf->mxsz]), (*m_hf ).h_contents[i], h_szs[i] * sizeof( uint32_t ));
    std::memcpy(&(bufBorders1D_host_ptr[i * hf->mxsz]), ( *m_hf ).h_borders[i], h_szs[i] * sizeof( uint32_t ));
  }
  alpaka::memcpy(queue,pImpl->bufContents1D_acc,bufContents1D_host);
  alpaka::memcpy(queue,pImpl->bufBorders1D_acc,bufBorders1D_host);
  
  m_hf_h      = hf;

  BufHostFHs bufFHs_host = alpaka::allocBuf<FHs, Idx>(alpaka::getDevByIdx<Host>(0u), Idx{1});
  FHs* bufFHs_host_ptr = alpaka::getPtrNative(bufFHs_host);
  *bufFHs_host_ptr = *m_hf_h;

  alpaka::memcpy(queue,pImpl->bufFHs_acc,bufFHs_host);
  alpaka::wait(queue);

  m_hf_d = alpaka::getPtrNative(pImpl->bufFHs_acc);
}
