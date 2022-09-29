/*
  Copyright (C) 2002-2021 CERN for the benefit of the ATLAS collaboration
*/

#include "gpuQ.h"
#include "DEV_BigMem.h"
#include <vector>

DEV_BigMem::DEV_BigMem( size_t s ) {  //initialize to one seg with size s
  void* p ;  
  m_seg_size = s ;
  gpuQ(cudaMalloc(&p , m_seg_size )) ;
  m_ptrs.push_back(p) ; 
  m_seg =0 ;
  m_used.push_back(0)  ;
} ; 

DEV_BigMem::~DEV_BigMem() {
  for(int i=0 ; i<m_ptrs.size() ; i++) gpuQ(cudaFree( m_ptrs[i]) ) ; 
}  ;


void *
DEV_BigMem::cu_bm_alloc(size_t s) {
  if (s  > (m_seg_size-m_used[m_seg]))  add_seg() ;
  long * q = (long *) m_ptrs[m_seg] ;
  int offset = m_used[m_seg]/sizeof(long) ;
  void * p = (void * )   &(q[offset])  ;
  m_used[m_seg] += ((s+sizeof(long)-1)/sizeof(long)  ) * sizeof(long)    ;
  return p  ;
	};

void DEV_BigMem::add_seg() { 
  void * p ; 
  gpuQ(cudaMalloc((void**)&p , m_seg_size )) ;
  m_ptrs.push_back(p) ;
  m_seg++;
  m_used.push_back(0)  ;
};


