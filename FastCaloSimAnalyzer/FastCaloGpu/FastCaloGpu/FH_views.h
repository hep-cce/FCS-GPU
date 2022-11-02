/*
  Copyright (C) 2002-2021 CERN for the benefit of the ATLAS collaboration
*/

#ifndef FH_VIEWS_H
#define FH_VIEWS_H

#include "FH_structs.h"
#include <Kokkos_Core.hpp>

typedef Kokkos::MemoryTraits<Kokkos::Unmanaged> KMTU;

typedef struct FHs_v {

  FHs_v(){};
  FHs_v( FHs* f1_h, FHs* f1_d ) {
    s_MaxValue = Kokkos::View<uint32_t>( &f1_h->s_MaxValue );
    nhist      = Kokkos::View<unsigned int>( &f1_d->nhist );

    low_edge = Kokkos::View<float*>( f1_d->low_edge,
                                     f1_h->nhist + 1 );
    h_szs    = Kokkos::View<unsigned int*>( f1_d->h_szs,
                                            f1_h->nhist );

    max_sz = 0;
    for (unsigned int i=0; i< f1_h->nhist; ++i) {
      tot_sz += f1_h->h_szs[i];
      max_sz = std::max(f1_h->h_szs[i],max_sz);
    }

    d_contents1D = Kokkos::View<uint32_t*>( f1_d->d_contents1D,
                                            max_sz*f1_h->nhist );
//                                         tot_sz );
    d_borders1D  = Kokkos::View<float*>( f1_d->d_borders1D,
                                         max_sz*f1_h->nhist );
//                                       tot_sz );
  }

  unsigned int                      tot_sz{0};
  unsigned int                      max_sz{0};
  Kokkos::View<uint32_t, KMTU>      s_MaxValue{};
  Kokkos::View<unsigned int, KMTU>  nhist{};
  Kokkos::View<float*, KMTU>        low_edge{};
  Kokkos::View<unsigned int*, KMTU> h_szs{};
  Kokkos::View<uint32_t*, KMTU>     d_contents1D{};
  Kokkos::View<float*, KMTU>        d_borders1D{};

} FHs_v;

typedef struct FH2D_v {

  FH2D_v(){};
  FH2D_v( FH2D *f2_h, FH2D *f2_d ) {

    nbinsx = Kokkos::View<int>( &f2_d->nbinsx );
    nbinsy = Kokkos::View<int>( &f2_d->nbinsy );

    bordersx = Kokkos::View<float*>( f2_d->h_bordersx,
                                     f2_h->nbinsx+1 );
    bordersy = Kokkos::View<float*>( f2_d->h_bordersy,
                                     f2_h->nbinsy+1 );
    contents = Kokkos::View<float*>( f2_d->h_contents,
                                     f2_h->nbinsx *
                                     f2_h->nbinsy );
  }

  Kokkos::View<int> nbinsx;
  Kokkos::View<int> nbinsy;

  Kokkos::View<float*> bordersx;
  Kokkos::View<float*> bordersy;
  Kokkos::View<float*> contents;
} FH2D_v;

#endif
