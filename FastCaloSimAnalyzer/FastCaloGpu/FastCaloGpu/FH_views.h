#ifndef FH_VIEWS_H
#define FH_VIEWS_H

#include "FH_structs.h"
#include "Args.h"
#include <Kokkos_Core.hpp>

typedef struct FHs_v {

  FHs_v(){};
  FHs_v( Chain0_Args& arg ) {
    s_MaxValue = Kokkos::View<uint32_t>( &arg.fhs->s_MaxValue );
    nhist      = Kokkos::View<unsigned int>( &arg.fhs->nhist );
    mxsz       = Kokkos::View<unsigned int>( &arg.fhs->mxsz );

    low_edge = Kokkos::View<float*>( arg.fhs_h.low_edge, arg.fhs_h.nhist + 1 );
    h_szs    = Kokkos::View<unsigned int*>( arg.fhs_h.h_szs, arg.fhs_h.nhist );

    d_contents1D = Kokkos::View<uint32_t*>( arg.fhs_h.d_contents1D, arg.fhs_h.nhist * arg.fhs_h.mxsz );
    d_borders1D  = Kokkos::View<float*>( arg.fhs_h.d_borders1D, arg.fhs_h.nhist * arg.fhs_h.mxsz );
  }

  Kokkos::View<uint32_t>      s_MaxValue;
  Kokkos::View<unsigned int>  nhist;
  Kokkos::View<unsigned int>  mxsz;
  Kokkos::View<float*>        low_edge;
  Kokkos::View<unsigned int*> h_szs;
  Kokkos::View<uint32_t*>     d_contents1D;
  Kokkos::View<float*>        d_borders1D;

} FHs_v;

typedef struct FH2D_v {

  FH2D_v(){};
  FH2D_v( Chain0_Args& arg ) {

    nbinsx = Kokkos::View<int>( &arg.fh2d->nbinsx );
    nbinsy = Kokkos::View<int>( &arg.fh2d->nbinsy );

    bordersx = Kokkos::View<float*>( arg.fh2d_h.h_bordersx, arg.fh2d_h.nbinsx+1 );
    bordersy = Kokkos::View<float*>( arg.fh2d_h.h_bordersy, arg.fh2d_h.nbinsy+1 );
    contents = Kokkos::View<float*>( arg.fh2d_h.h_contents, arg.fh2d_h.nbinsx * arg.fh2d_h.nbinsy );
  }

  Kokkos::View<int> nbinsx;
  Kokkos::View<int> nbinsy;

  Kokkos::View<float*> bordersx;
  Kokkos::View<float*> bordersy;
  Kokkos::View<float*> contents;
} FH2D_v;

#endif
