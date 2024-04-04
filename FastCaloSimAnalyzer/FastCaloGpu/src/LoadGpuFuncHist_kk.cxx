/*
  Copyright (C) 2002-2021 CERN for the benefit of the ATLAS collaboration
*/

#include "LoadGpuFuncHist.h"
#include <iostream>

LoadGpuFuncHist::~LoadGpuFuncHist() {
  delete m_hf;
  delete m_hf_h;

  delete m_hf_v;
  delete m_hf2d_v;

  // FIXME: need to delete all the internally allocated views too
}

LoadGpuFuncHist::LoadGpuFuncHist() {}

void LoadGpuFuncHist::LD2D() {
  if ( !m_hf2d ) {
    std::cout << "Error Load 2DFunctionHisto " << std::endl;
    return;
  }

  // FIXME - this leaks memory as it's called twice per event, but
  // the device mem and pointers are not deleted.

  FH2D* hf_ptr = new FH2D;

  m_hf2d_v         = new FH2D_v;
  m_hf2d_v->nbinsx = Kokkos::View<int>( "nbinsx" );
  m_hf2d_v->nbinsy = Kokkos::View<int>( "nbinsy" );

  hf_ptr->nbinsx = m_hf2d->nbinsx;
  hf_ptr->nbinsy = m_hf2d->nbinsy;

  Kokkos::View<int, Kokkos::HostSpace> nbx( &m_hf2d->nbinsx );
  Kokkos::View<int, Kokkos::HostSpace> nby( &m_hf2d->nbinsy );

  Kokkos::deep_copy( m_hf2d_v->nbinsx, nbx );
  Kokkos::deep_copy( m_hf2d_v->nbinsy, nby );

  m_hf2d_v->bordersx = Kokkos::View<float*>( "bordersx", hf_ptr->nbinsx + 1 );
  m_hf2d_v->bordersy = Kokkos::View<float*>( "bordersy", hf_ptr->nbinsy + 1 );
  m_hf2d_v->contents = Kokkos::View<float*>( "contents", hf_ptr->nbinsy * hf_ptr->nbinsx );
  Kokkos::View<float*, Kokkos::HostSpace> bx( m_hf2d->h_bordersx, hf_ptr->nbinsx + 1 );
  Kokkos::View<float*, Kokkos::HostSpace> by( m_hf2d->h_bordersy, hf_ptr->nbinsy + 1 );
  Kokkos::View<float*, Kokkos::HostSpace> ct( m_hf2d->h_contents, hf_ptr->nbinsx * hf_ptr->nbinsy );

  Kokkos::deep_copy( m_hf2d_v->bordersx, bx );
  Kokkos::deep_copy( m_hf2d_v->bordersy, by );
  Kokkos::deep_copy( m_hf2d_v->contents, ct );

  hf_ptr->h_bordersx = m_hf2d_v->bordersx.data();
  hf_ptr->h_bordersy = m_hf2d_v->bordersy.data();
  hf_ptr->h_contents = m_hf2d_v->contents.data();

  m_hf2d_h = hf_ptr;

  m_hf2d_dv = Kokkos::View<FH2D>( "hf2d" );
  Kokkos::View<FH2D, Kokkos::HostSpace> hfv( m_hf2d_h );
  Kokkos::deep_copy( m_hf2d_dv, hfv );
  m_hf2d_d = m_hf2d_dv.data();
}

void LoadGpuFuncHist::LD() {
  // this call  assume  already have Histofuncs set in m_hf
  // this function allocate memory of GPU and deep copy m_hf to m_hf_d

  if ( !m_hf ) {
    std::cout << "Error Load WiggleHistoFunctions " << std::endl;
    return;
  }

  FHs* hf_ptr = new FHs;

  m_hf_v = new FHs_v;

  hf_ptr->s_MaxValue  = m_hf->s_MaxValue;
  hf_ptr->nhist       = m_hf->nhist;
  hf_ptr->mxsz        = m_hf->mxsz;
  unsigned int* h_szs = m_hf->h_szs; // already allocateded on host ;

  m_hf_v->low_edge     = Kokkos::View<float*>( "low edge", hf_ptr->nhist + 1 );
  m_hf_v->h_szs        = Kokkos::View<unsigned int*>( "hist sizes", hf_ptr->nhist );
  m_hf_v->d_contents1D = Kokkos::View<uint32_t*>( "contents", hf_ptr->nhist * hf_ptr->mxsz );
  m_hf_v->d_borders1D  = Kokkos::View<float*>( "borders", hf_ptr->nhist * hf_ptr->mxsz );

  Kokkos::View<float*, Kokkos::HostSpace>        le( m_hf->low_edge, hf_ptr->nhist + 1 );
  Kokkos::View<unsigned int*, Kokkos::HostSpace> sz( m_hf->h_szs, hf_ptr->nhist );

  Kokkos::deep_copy( m_hf_v->low_edge, le );
  Kokkos::deep_copy( m_hf_v->h_szs, sz );

  for ( size_t i = 0; i < hf_ptr->nhist; ++i ) {
    Kokkos::View<uint32_t*, Kokkos::HostSpace> ct( m_hf->h_contents[i], h_szs[i] );
    Kokkos::View<float*, Kokkos::HostSpace>    bd( m_hf->h_borders[i], h_szs[i] );

    auto c_sub =
        Kokkos::subview( m_hf_v->d_contents1D, std::make_pair( i * hf_ptr->mxsz, ( i * hf_ptr->mxsz ) + h_szs[i] ) );
    auto b_sub =
        Kokkos::subview( m_hf_v->d_borders1D, std::make_pair( i * hf_ptr->mxsz, ( i * hf_ptr->mxsz ) + h_szs[i] ) );

    Kokkos::deep_copy( c_sub, ct );
    Kokkos::deep_copy( b_sub, bd );
  }

  hf_ptr->low_edge     = m_hf_v->low_edge.data();
  hf_ptr->h_szs        = m_hf_v->h_szs.data();
  hf_ptr->d_contents1D = m_hf_v->d_contents1D.data();
  hf_ptr->d_borders1D  = m_hf_v->d_borders1D.data();

  m_hf_h = hf_ptr;

  m_hf_dv = Kokkos::View<FHs>( "hfs" );
  Kokkos::View<FHs, Kokkos::HostSpace> hfv( m_hf_h );
  Kokkos::deep_copy( m_hf_dv, hfv );
  m_hf_d = m_hf_dv.data();
}
