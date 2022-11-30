/*
  Copyright (C) 2002-2021 CERN for the benefit of the ATLAS collaboration
*/

#ifndef DEV_BIGMEM_H
#define DEV_BIGMEM_H

#include <vector>
#include <iostream>

class DEV_BigMem {

public:
  DEV_BigMem() : m_ptrs( 0 ), m_seg( 0 ), m_seg_size( 0 ), m_used( 0 ){};
  DEV_BigMem( size_t s );
  ~DEV_BigMem();

  void* dev_bm_alloc( size_t s ) {
    if ( s > ( m_seg_size - m_used[m_seg] ) ) add_seg();
    long* q      = (long*)m_ptrs[m_seg];
    int   offset = m_used[m_seg] / sizeof( long );
    void* p      = (void*)&( q[offset] );
    m_used[m_seg] += ( ( s + sizeof( long ) - 1 ) / sizeof( long ) ) * sizeof( long );
    // std::cout << "BM alloc size " << s << "  total " << m_used[m_seg]
    //           << " in seg " << m_seg << std::endl;

    return p;
  };

  size_t size() { return ( m_seg + 1 ) * m_seg_size; };
  size_t used() { return m_seg * m_seg_size + m_used[m_seg]; };

  static DEV_BigMem* bm_ptr;

private:
  void add_seg();

  std::vector<void*>  m_ptrs;          // points to each allocated segment
  int                 m_seg{ 0 };      // do we need ? it's current size of mptrs -1 ;
  size_t              m_seg_size{ 0 }; // size of each allocation
  std::vector<size_t> m_used;          // used memory in each allocated segment
};

#endif
