/*
  Copyright (C) 2002-2021 CERN for the benefit of the ATLAS collaboration
*/

#ifndef DEV_BIGMEM_H
#define DEV_BIGMEM_H


#include <vector>

class DEV_BigMem  {

public :
  DEV_BigMem(): m_seg_size(0), m_ptrs(0), m_used(0), m_seg(0)   { } ; 
  DEV_BigMem( size_t s ) ;
  ~DEV_BigMem() ;

  void * cu_bm_alloc(size_t s) ;

  size_t  size() { return (m_seg+1) * m_seg_size ; } ;
  size_t  used() { return m_seg * m_seg_size + m_used[m_seg] ; };

  static DEV_BigMem * bm_ptr ;

private : 
  void add_seg();
		
  std::vector<void*>  m_ptrs  ;   //points to each allocated segment 
  int  m_seg ;                    //do we need ? it's current size of mptrs -1 ;
  size_t   m_seg_size  ;          // size of each allocation
  std::vector < size_t>  m_used ; // used memory in each allocated segment

} ;




#endif
