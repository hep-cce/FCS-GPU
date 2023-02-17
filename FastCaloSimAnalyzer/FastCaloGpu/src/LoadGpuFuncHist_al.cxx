/*
  Copyright (C) 2002-2021 CERN for the benefit of the ATLAS collaboration
*/

#include "LoadGpuFuncHist.h"
#include "AlpakaDefs.h"
#include <alpaka/alpaka.hpp>
#include <iostream>
#include "DEV_BigMem.h"

DEV_BigMem *DEV_BigMem::bm_ptr;

LoadGpuFuncHist::LoadGpuFuncHist() 
{
}

LoadGpuFuncHist::~LoadGpuFuncHist() {
  free(m_hf);
  cudaFree(m_hf_h);

  free(m_hf2d);
  free(m_hf2d_h);
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

  DEV_BigMem *p = DEV_BigMem::bm_ptr;

  m_hf2d_h->h_bordersx = p->dev_bm_alloc<float>((m_hf2d_h->nbinsx + 1));
  m_hf2d_h->h_bordersy = p->dev_bm_alloc<float>((m_hf2d_h->nbinsy + 1));
  m_hf2d_h->h_contents = p->dev_bm_alloc<float>(m_hf2d_h->nbinsy * m_hf2d_h->nbinsx);

  auto h_bordersXView = alpaka::createView(alpaka::getDevByIdx<Host>(0u), (*m_hf2d).h_bordersx, static_cast<Idx>(( m_hf2d_h->nbinsx + 1 )*sizeof(float)));
  auto d_bordersXView = alpaka::createView(alpaka::getDevByIdx<Acc>(0u),m_hf2d_h->h_bordersx, static_cast<Idx>(( m_hf2d_h->nbinsx + 1 )*sizeof(float)));

  auto h_bordersYView = alpaka::createView(alpaka::getDevByIdx<Host>(0u), (*m_hf2d).h_bordersy, static_cast<Idx>(( m_hf2d_h->nbinsy + 1 )*sizeof(float)));
  auto d_bordersYView = alpaka::createView(alpaka::getDevByIdx<Acc>(0u),m_hf2d_h->h_bordersy, static_cast<Idx>(( m_hf2d_h->nbinsy + 1 )*sizeof(float)));

  auto h_contentsView = alpaka::createView(alpaka::getDevByIdx<Host>(0u), (*m_hf2d).h_contents, static_cast<Idx>((m_hf2d_h->nbinsx * m_hf2d_h->nbinsy) * sizeof(float)));
  auto d_contentsView = alpaka::createView(alpaka::getDevByIdx<Acc>(0u),m_hf2d_h->h_contents, static_cast<Idx>((m_hf2d_h->nbinsy * m_hf2d_h->nbinsx) * sizeof(float)));

  m_hf2d_d = p->dev_bm_alloc<FH2D>(1);
  auto h_hf2dView = alpaka::createView(alpaka::getDevByIdx<Host>(0u),m_hf2d_h,static_cast<Idx>(sizeof(FH2D)));
  auto d_hf2dView = alpaka::createView(alpaka::getDevByIdx<Acc>(0u),m_hf2d_d,static_cast<Idx>(sizeof(FH2D)));

  QueueAcc queue(alpaka::getDevByIdx<Acc>(Idx{0}));
  alpaka::memcpy(queue,d_bordersXView,h_bordersXView);
  alpaka::memcpy(queue,d_bordersYView,h_bordersYView);
  alpaka::memcpy(queue,d_contentsView,h_contentsView);
  alpaka::memcpy(queue,d_hf2dView,h_hf2dView);
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

  DEV_BigMem *p = DEV_BigMem::bm_ptr;

  hf.low_edge = p->dev_bm_alloc<float>((hf.nhist + 1));
  hf.h_szs = p->dev_bm_alloc<unsigned int>(hf.nhist);

  auto h_lowEdgeView = alpaka::createView(alpaka::getDevByIdx<Host>(0u), (*m_hf).low_edge, static_cast<Idx>(( hf.nhist + 1 )*sizeof(float)));
  auto d_lowEdgeView = alpaka::createView(alpaka::getDevByIdx<Acc>(0u), hf.low_edge, static_cast<Idx>(( hf.nhist + 1 )*sizeof(float)));

  auto h_szsView = alpaka::createView(alpaka::getDevByIdx<Host>(0u),(*m_hf).h_szs, static_cast<Idx>(hf.nhist * sizeof(unsigned int)));
  auto d_szsView = alpaka::createView(alpaka::getDevByIdx<Acc>(0u), hf.h_szs, static_cast<Idx>(hf.nhist * sizeof(unsigned int)));

  QueueAcc queue(alpaka::getDevByIdx<Acc>(Idx{0}));
  alpaka::memcpy(queue,d_lowEdgeView,h_lowEdgeView);
  alpaka::memcpy(queue,d_szsView,h_szsView);

  hf.h_contents = p->dev_bm_alloc<uint32_t *>(hf.nhist);
  hf.h_borders = p->dev_bm_alloc<float *>(hf.nhist);

  uint32_t **contents_ptr = (uint32_t **)malloc(hf.nhist * sizeof(uint32_t *));
  float **borders_ptr = (float **)malloc(hf.nhist * sizeof(float *));

  for (unsigned int i = 0; i < hf.nhist; ++i) {
    contents_ptr[i] = p->dev_bm_alloc<uint32_t>(h_szs[i]);
    borders_ptr[i] = p->dev_bm_alloc<float>((h_szs[i] + 1));

    auto h_contentsView = alpaka::createView(alpaka::getDevByIdx<Host>(0u), (*m_hf).h_contents[i], static_cast<Idx>(h_szs[i] * sizeof(uint32_t)));
    auto d_contentsView = alpaka::createView(alpaka::getDevByIdx<Acc>(0u), contents_ptr[i], static_cast<Idx>(h_szs[i] * sizeof(uint32_t)));

    auto h_bordersView = alpaka::createView(alpaka::getDevByIdx<Host>(0u), (*m_hf).h_borders[i], static_cast<Idx>((h_szs[i] + 1) * sizeof(float)));
    auto d_bordersView = alpaka::createView(alpaka::getDevByIdx<Acc>(0u), borders_ptr[i], static_cast<Idx>((h_szs[i] + 1) * sizeof(float)));

    alpaka::memcpy(queue,d_contentsView,h_contentsView);
    alpaka::memcpy(queue,d_bordersView,h_bordersView);
  }

  auto h_contentView = alpaka::createView(alpaka::getDevByIdx<Host>(0u), contents_ptr, static_cast<Idx>(hf.nhist * sizeof(uint32_t*)));
  auto d_contentView = alpaka::createView(alpaka::getDevByIdx<Acc>(0u), hf.h_contents, static_cast<Idx>(hf.nhist * sizeof(uint32_t*)));

  auto h_borderView = alpaka::createView(alpaka::getDevByIdx<Host>(0u), borders_ptr, static_cast<Idx>(hf.nhist * sizeof(float*)));
  auto d_borderView = alpaka::createView(alpaka::getDevByIdx<Acc>(0u), hf.h_borders, static_cast<Idx>(hf.nhist * sizeof(float*)));

  alpaka::memcpy(queue,d_contentView,h_contentView);
  alpaka::memcpy(queue,d_borderView,h_borderView);

  m_hf_d = p->dev_bm_alloc<FHs>(1);

  auto h_hfView = alpaka::createView(alpaka::getDevByIdx<Host>(0u), &hf, static_cast<Idx>(sizeof(FHs)));
  auto d_hfView = alpaka::createView(alpaka::getDevByIdx<Acc>(0u), m_hf_d, static_cast<Idx>(sizeof(FHs)));
  alpaka::memcpy(queue,d_hfView,h_hfView);

  alpaka::wait(queue);

  free(contents_ptr);
  free(borders_ptr);

  m_hf_h = &hf;
}
