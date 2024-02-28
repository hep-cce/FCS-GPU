/*
  Copyright (C) 2002-2021 CERN for the benefit of the ATLAS collaboration
*/

#include "LoadGpuFuncHist.h"
#include "AlpakaDefs.h"
#include <alpaka/alpaka.hpp>
#include <iostream>
#include <vector>

class LoadGpuFuncHist::Impl {
public:
  Impl()
    : bufBordersX_acc{alpaka::allocBuf<float, Idx>(alpaka::getDevByIdx<Acc>(0u), Idx{1})}
    , bufBordersY_acc{alpaka::allocBuf<float, Idx>(alpaka::getDevByIdx<Acc>(0u), Idx{1})}
    , bufContents_acc{alpaka::allocBuf<float, Idx>(alpaka::getDevByIdx<Acc>(0u), Idx{1})}
    , bufFH2D_acc{alpaka::allocBuf<FH2D, Idx>(alpaka::getDevByIdx<Acc>(0u), Idx{1})}
    , bufLowEdge_acc{alpaka::allocBuf<float, Idx>(alpaka::getDevByIdx<Acc>(0u), Idx{1})}
    , bufSzs_acc{alpaka::allocBuf<unsigned, Idx>(alpaka::getDevByIdx<Acc>(0u), Idx{1})}
    , bufContentsPtr_acc{alpaka::allocBuf<uint32_t*, Idx>(alpaka::getDevByIdx<Acc>(0u), Idx{1})}
    , bufBordersPtr_acc{alpaka::allocBuf<float*, Idx>(alpaka::getDevByIdx<Acc>(0u), Idx{1})}
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

  std::vector<BufAccUint32> bufContents1D_acc;
  std::vector<BufAcc> bufBorders1D_acc;
  BufAcctUint32Ptr bufContentsPtr_acc;
  BufAccFloatPtr bufBordersPtr_acc;

  BufAccFHs bufFHs_acc;
};



LoadGpuFuncHist::LoadGpuFuncHist() 
{
  pImpl = new Impl();
}

LoadGpuFuncHist::~LoadGpuFuncHist() {
  free(m_hf);

  free(m_hf2d);
  free(m_hf2d_h);

  delete pImpl;
}

/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
void LoadGpuFuncHist::LD2D() {
  if (!m_hf2d) {
    std::cout << "Error Load 2DFunctionHisto " << std::endl;
    return;
  }

  m_hf2d_h = new FH2D;

  m_hf2d_h->nbinsx = (*m_hf2d).nbinsx;
  m_hf2d_h->nbinsy = (*m_hf2d).nbinsy;

  pImpl->bufBordersX_acc = alpaka::allocBuf<float, Idx>(alpaka::getDevByIdx<Acc>(0u), static_cast<Idx>(m_hf2d_h->nbinsx + 1));
  pImpl->bufBordersY_acc = alpaka::allocBuf<float, Idx>(alpaka::getDevByIdx<Acc>(0u), static_cast<Idx>(m_hf2d_h->nbinsy + 1));
  pImpl->bufContents_acc = alpaka::allocBuf<float, Idx>(alpaka::getDevByIdx<Acc>(0u), static_cast<Idx>(m_hf2d_h->nbinsy * m_hf2d_h->nbinsx));

  m_hf2d_h->h_bordersx = alpaka::getPtrNative(pImpl->bufBordersX_acc);
  m_hf2d_h->h_bordersy = alpaka::getPtrNative(pImpl->bufBordersY_acc);
  m_hf2d_h->h_contents = alpaka::getPtrNative(pImpl->bufContents_acc);

  BufHost bufBordersX_host = alpaka::allocBuf<float, Idx>(alpaka::getDevByIdx<Host>(0u), static_cast<Idx>(m_hf2d_h->nbinsx + 1));
  float* bufBordersX_host_ptr = alpaka::getPtrNative(bufBordersX_host);
  std::memcpy(bufBordersX_host_ptr,(*m_hf2d).h_bordersx,( m_hf2d_h->nbinsx + 1 )*sizeof(float));

  BufHost bufBordersY_host = alpaka::allocBuf<float, Idx>(alpaka::getDevByIdx<Host>(0u), static_cast<Idx>(m_hf2d_h->nbinsy + 1));
  float* bufBordersY_host_ptr = alpaka::getPtrNative(bufBordersY_host);
  std::memcpy(bufBordersY_host_ptr,(*m_hf2d).h_bordersy,( m_hf2d_h->nbinsy + 1 )*sizeof(float));

  BufHost bufContents_host = alpaka::allocBuf<float, Idx>(alpaka::getDevByIdx<Host>(0u), static_cast<Idx>(m_hf2d_h->nbinsy * m_hf2d_h->nbinsx));
  float* bufContents_host_ptr = alpaka::getPtrNative(bufContents_host);
  std::memcpy(bufContents_host_ptr,(*m_hf2d).h_contents,(m_hf2d_h->nbinsx * m_hf2d_h->nbinsy) * sizeof(float));

  m_hf2d_d = alpaka::getPtrNative(pImpl->bufFH2D_acc);
  BufHostFH2D bufFH2D_host = alpaka::allocBuf<FH2D, Idx>(alpaka::getDevByIdx<Host>(0u), Idx{1});
  FH2D* bufFH2D_host_ptr = alpaka::getPtrNative(bufFH2D_host);
  *bufFH2D_host_ptr = *m_hf2d_h;

  QueueAcc queue(alpaka::getDevByIdx<Acc>(Idx{0}));
  alpaka::memcpy(queue,pImpl->bufBordersX_acc,bufBordersX_host);
  alpaka::memcpy(queue,pImpl->bufBordersY_acc,bufBordersY_host);
  alpaka::memcpy(queue,pImpl->bufContents_acc,bufContents_host);
  alpaka::memcpy(queue,pImpl->bufFH2D_acc,bufFH2D_host);
  alpaka::wait(queue);
}

/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

void LoadGpuFuncHist::LD() {
  // this call  assume  already have Histofuncs set in m_hf
  // this function allocate memory of GPU and deep copy m_hf to m_hf_h

  if (!m_hf) {
    std::cout << "Error Load WiggleHistoFunctions " << std::endl;
    return;
  }

  FHs hf = { 0, 0, 0, 0, 0, 0 };
  hf.s_MaxValue = (*m_hf).s_MaxValue;
  hf.nhist = (*m_hf).nhist;
  unsigned int *h_szs = (*m_hf).h_szs; // already allocateded on host ;

  QueueAcc queue(alpaka::getDevByIdx<Acc>(Idx{0}));

  pImpl->bufLowEdge_acc = alpaka::allocBuf<float, Idx>(alpaka::getDevByIdx<Acc>(0u), static_cast<Idx>( hf.nhist + 1 ));
  pImpl->bufSzs_acc = alpaka::allocBuf<unsigned, Idx>(alpaka::getDevByIdx<Acc>(0u), static_cast<Idx>( hf.nhist ));

  hf.low_edge= alpaka::getPtrNative(pImpl->bufLowEdge_acc);
  hf.h_szs = alpaka::getPtrNative(pImpl->bufSzs_acc);

  BufHost bufLowEdge_host = alpaka::allocBuf<float, Idx>(alpaka::getDevByIdx<Host>(0u), static_cast<Idx>( hf.nhist + 1 ));
  float* bufLowEdge_host_ptr = alpaka::getPtrNative(bufLowEdge_host);
  std::memcpy(bufLowEdge_host_ptr,( *m_hf ).low_edge, ( hf.nhist + 1 ) * sizeof( float ));

  BufHostUnsigned bufSzs_host = alpaka::allocBuf<unsigned, Idx>(alpaka::getDevByIdx<Host>(0u), static_cast<Idx>( hf.nhist ));
  unsigned* bufSzs_host_ptr = alpaka::getPtrNative(bufSzs_host);
  std::memcpy(bufSzs_host_ptr, ( *m_hf ).h_szs, hf.nhist * sizeof( unsigned int ));

  alpaka::memcpy(queue,pImpl->bufLowEdge_acc,bufLowEdge_host);
  alpaka::memcpy(queue,pImpl->bufSzs_acc,bufSzs_host);
  alpaka::wait(queue);

  pImpl->bufContentsPtr_acc = alpaka::allocBuf<uint32_t*, Idx>(alpaka::getDevByIdx<Acc>(0u), static_cast<Idx>(hf.nhist));
  hf.h_contents = alpaka::getPtrNative(pImpl->bufContentsPtr_acc);
  BufHostUint32Ptr bufContentsPtr_host = alpaka::allocBuf<uint32_t*, Idx>(alpaka::getDevByIdx<Host>(0u), static_cast<Idx>(hf.nhist));
  uint32_t **contents_ptr = alpaka::getPtrNative(bufContentsPtr_host);

  pImpl->bufBordersPtr_acc = alpaka::allocBuf<float*, Idx>(alpaka::getDevByIdx<Acc>(0u), static_cast<Idx>(hf.nhist));
  hf.h_borders = alpaka::getPtrNative(pImpl->bufBordersPtr_acc);
  BufHostFloatPtr bufBordersPtr_host = alpaka::allocBuf<float*, Idx>(alpaka::getDevByIdx<Host>(0u), static_cast<Idx>(hf.nhist));
  float **borders_ptr = alpaka::getPtrNative(bufBordersPtr_host);

  for (unsigned int i = 0; i < hf.nhist; ++i) {

    BufAccUint32 bufContents_acc = alpaka::allocBuf<uint32_t, Idx>(alpaka::getDevByIdx<Acc>(0u), static_cast<Idx>(h_szs[i]));
    contents_ptr[i] = alpaka::getPtrNative(bufContents_acc);
    pImpl->bufContents1D_acc.push_back(bufContents_acc);
    BufHostUint32 bufContents_host = alpaka::allocBuf<uint32_t, Idx>(alpaka::getDevByIdx<Host>(0u), static_cast<Idx>(h_szs[i]));
    uint32_t* bufContents_host_ptr = alpaka::getPtrNative(bufContents_host);
    std::memcpy(bufContents_host_ptr,(*m_hf).h_contents[i],h_szs[i] * sizeof(uint32_t));
    alpaka::memcpy(queue,bufContents_acc,bufContents_host);

    BufAcc bufBorders_acc = alpaka::allocBuf<float, Idx>(alpaka::getDevByIdx<Acc>(0u), static_cast<Idx>(h_szs[i] + 1));
    borders_ptr[i] = alpaka::getPtrNative(bufBorders_acc);
    pImpl->bufBorders1D_acc.push_back(bufBorders_acc);
    BufHost bufBorders_host = alpaka::allocBuf<float, Idx>(alpaka::getDevByIdx<Host>(0u), static_cast<Idx>(h_szs[i] + 1));
    float* bufBorders_host_ptr = alpaka::getPtrNative(bufBorders_host);
    std::memcpy(bufBorders_host_ptr,(*m_hf).h_borders[i],(h_szs[i] + 1) * sizeof(float));
    alpaka::memcpy(queue,bufBorders_acc,bufBorders_host);

    alpaka::wait(queue);
  }

  alpaka::memcpy(queue,pImpl->bufContentsPtr_acc,bufContentsPtr_host);
  alpaka::memcpy(queue,pImpl->bufBordersPtr_acc,bufBordersPtr_host);
  alpaka::wait(queue);

  m_hf_d = alpaka::getPtrNative(pImpl->bufFHs_acc);
  BufHostFHs bufFHs_host = alpaka::allocBuf<FHs, Idx>(alpaka::getDevByIdx<Host>(0u), Idx{1});
  FHs* bufFHs_host_ptr = alpaka::getPtrNative(bufFHs_host);
  std::memcpy(bufFHs_host_ptr,&hf,sizeof(FHs));
  alpaka::memcpy(queue,pImpl->bufFHs_acc,bufFHs_host);
  alpaka::wait(queue);

  m_hf_h = &hf;
}
