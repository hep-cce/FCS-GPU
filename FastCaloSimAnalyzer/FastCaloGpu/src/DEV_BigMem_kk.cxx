#include "DEV_BigMem.h"
#include <vector>
#include <Kokkos_Core.hpp>

DEV_BigMem::DEV_BigMem( size_t s ) {  //initialize to one seg with size s
  m_seg_size = s ;

  void* p {nullptr};
  try {
    p = Kokkos::kokkos_malloc("bigmem buffer", m_seg_size);
  } catch (...) {
    std::cerr << "unable to allocate " << m_seg_size << " bytes for bigmem buffer\n";
    return;
  }
  m_ptrs.push_back(p);
  m_seg = 0;
  m_used.push_back(0);
}

DEV_BigMem::~DEV_BigMem() {
  for(long unsigned int i=0 ; i<m_ptrs.size() ; i++) {
    Kokkos::kokkos_free( m_ptrs[i] );
  }
} 


void DEV_BigMem::add_seg() {

  void* p {nullptr};
  try {
    p = Kokkos::kokkos_malloc("bigmem buffer", m_seg_size);
  } catch (...) {
    std::cerr << "unable to allocate " << m_seg_size << " bytes for bigmem buffer\n";
    return;
  }
  std::cout << "DEV_BM add_seg() " << m_seg << " " << p << std::endl;
  m_ptrs.push_back(p) ;
  m_seg++;
  m_used.push_back(0);

};

