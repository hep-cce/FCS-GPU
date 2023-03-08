/*
  Copyright (C) 2002-2023 CERN for the benefit of the ATLAS collaboration
*/

#include "DEV_BigMem.h"

DEV_BigMem::DEV_BigMem(size_t s) { // initialize to one seg with size s
  m_seg_size = s;
  BufAccChar buf = alpaka::allocBuf<char, Idx>(alpaka::getDevByIdx<Acc>(0u),Vec{Idx(s)});
  m_bufs.push_back(buf);
  m_ptrs.push_back(alpaka::getPtrNative(buf));
  m_seg = 0;
  m_used.push_back(0);
}

DEV_BigMem::~DEV_BigMem() {
  m_bufs.clear();
}

void DEV_BigMem::add_seg() {
  BufAccChar buf = alpaka::allocBuf<char, Idx>(alpaka::getDevByIdx<Acc>(0u),Vec{Idx(m_seg_size)});
  m_bufs.push_back(buf);
  m_ptrs.push_back(alpaka::getPtrNative(buf));
  m_seg++;
  m_used.push_back(0);
};
