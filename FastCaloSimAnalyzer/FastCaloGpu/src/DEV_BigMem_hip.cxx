/*
  Copyright (C) 2002-2021 CERN for the benefit of the ATLAS collaboration
*/

#include "gpuQ.h"
#include "DEV_BigMem.h"
#include <vector>

DEV_BigMem::DEV_BigMem(size_t s) { // initialize to one seg with size s
  void *p;
  m_seg_size = s;
  gpuQ(hipMalloc(&p, m_seg_size));
  m_ptrs.push_back(p);
  m_seg = 0;
  m_used.push_back(0);
};

DEV_BigMem::~DEV_BigMem() {
  for (int i = 0; i < m_ptrs.size(); i++)
    gpuQ(hipFree(m_ptrs[i]));
};

void DEV_BigMem::add_seg() {
  void *p;
  gpuQ(hipMalloc((void **)&p, m_seg_size));
  m_ptrs.push_back(p);
  m_seg++;
  m_used.push_back(0);
};
