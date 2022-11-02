/*
  Copyright (C) 2002-2021 CERN for the benefit of the ATLAS collaboration
*/

#include "LoadGpuFuncHist.h"
#include "DEV_BigMem.h"
#include <iostream>

DEV_BigMem* DEV_BigMem::bm_ptr;

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


  std::cout << "LD2D -----\n";
  FH2D_v ff = *m_hf2d_v;
  Kokkos::parallel_for(
                       1, KOKKOS_LAMBDA( int i ) {
                         printf("%d %d",i,ff.nbinsx());
                         for ( int j=0; j< ff.nbinsx(); ++j) {
                           printf(" %f",ff.bordersx[j]);
                           }
                         printf("\n");  
                         }
                         );
  Kokkos::fence();
  

}

/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

void LoadGpuFuncHist::LD() {
  // this call  assume  already have Histofuncs set in m_hf
  // this function allocate memory of GPU and deep copy m_hf to m_hf_d

  if ( !m_hf ) {
    std::cout << "Error Load WiggleHistoFunctions " << std::endl;
    return;
  }

  FHs* hf_ptr = new FHs;             // on host with device ptrs

  m_hf_v = new FHs_v;                // View on host with device ptrs

  hf_ptr->s_MaxValue  = m_hf->s_MaxValue;
  hf_ptr->nhist       = m_hf->nhist;
  unsigned int* h_szs = m_hf->h_szs; // already allocateded on host ;

  DEV_BigMem* p = DEV_BigMem::bm_ptr;

  //  std::cout << "DEV: " << p << p->size() << " " << p->used() << std::endl;
  
  //  unsigned int sz{0};
  //  sz = hf_ptr->nhist * sizeof(float);


  // Kokkos::View<int*, Kokkos::HostSpace> vh("test",10);
  // for (int i=0; i<10; ++i) {
  //     vh(i) = i;
  //     std::cout << "--> vh " << i << ": " << vh(i) << std::endl;
  // }
  // void* bm = p->dev_bm_alloc( sizeof(int)*10 );
  // std::cout << "bm: " << bm << " sz: " << p->size() << "  us: " << p->used() << std::endl;
  // Kokkos::View<int*, KMTU> vd ( (int*) p->dev_bm_alloc( sizeof(int)*10), 10  );

  // Kokkos::deep_copy(vd, vh);

  // Kokkos::parallel_for( 10, KOKKOS_LAMBDA(int tid ) {
  //    vd(tid) = vd(tid) + 100;
  //    printf("  --> %d %d\n",tid,vd(tid));
  //    } );

  // Kokkos::View<int*, Kokkos::HostSpace> vh2("test2",10);
  // Kokkos::deep_copy(vh2, vd);
  // for (int i=0; i<10; ++i) {
  //    std::cout << "===?> vh2 " << i << " " << vh2(i) << std::endl;
  // }

  Kokkos::View<unsigned int, Kokkos::HostSpace> vnh(&hf_ptr->nhist);
  m_hf_v->nhist = Kokkos::View<unsigned int, KMTU>
     ( (unsigned int*) p->dev_bm_alloc( sizeof(unsigned int) ), 1 );
  Kokkos::deep_copy(m_hf_v->nhist, vnh);

  Kokkos::View<uint32_t, Kokkos::HostSpace> vmx( &hf_ptr->s_MaxValue );
  m_hf_v->s_MaxValue = Kokkos::View<uint32_t, KMTU> (
     (uint32_t*) p->dev_bm_alloc( sizeof(uint32_t) ), 1);
  Kokkos::deep_copy(m_hf_v->s_MaxValue, vmx);

  
  m_hf_v->low_edge = Kokkos::View<float*, KMTU >
    ( (float*) p->dev_bm_alloc( (hf_ptr->nhist+1)*sizeof(float)),
      hf_ptr->nhist+1 );
  
  m_hf_v->h_szs = Kokkos::View<unsigned int*, KMTU >
    ( (unsigned int*) p->dev_bm_alloc( (hf_ptr->nhist)*sizeof(unsigned int)),
      hf_ptr->nhist );

  unsigned int mxsz{0};
  unsigned int tot_sz{0};
  for (unsigned int i=0; i< hf_ptr->nhist; ++i) {
    mxsz = std::max(m_hf->h_szs[i], mxsz);
    tot_sz += m_hf->h_szs[i];
  }
  m_hf_v->tot_sz = tot_sz;
  m_hf_v->max_sz = mxsz;
  
  m_hf_v->d_contents1D = Kokkos::View<uint32_t*, KMTU >
    ( (uint32_t*) p->dev_bm_alloc( (hf_ptr->nhist)*mxsz*sizeof(uint32_t) ),
      hf_ptr->nhist * mxsz );


  m_hf_v->d_borders1D = Kokkos::View<float*, KMTU >
    ( (float*) p->dev_bm_alloc( (hf_ptr->nhist)*mxsz*sizeof(float) ),
      hf_ptr->nhist * mxsz );


  Kokkos::View<float*, Kokkos::HostSpace>        le( m_hf->low_edge, hf_ptr->nhist + 1 );
  Kokkos::View<unsigned int*, Kokkos::HostSpace> sz( m_hf->h_szs, hf_ptr->nhist );

  Kokkos::deep_copy( m_hf_v->low_edge, le );

  // Kokkos::View<float*> kv = m_hf_v->low_edge;
  // Kokkos::parallel_for( hf_ptr->nhist+1, KOKKOS_LAMBDA(int tid ) {
  //     printf("le: %d %f\n",tid,kv(tid));
  //     } );
  
  Kokkos::deep_copy( m_hf_v->h_szs, sz );

  for ( size_t i = 0; i < hf_ptr->nhist; ++i ) {
    Kokkos::View<uint32_t*, Kokkos::HostSpace> ct( m_hf->h_contents[i], h_szs[i] );
    Kokkos::View<float*, Kokkos::HostSpace>    bd( m_hf->h_borders[i], h_szs[i] );

    auto c_sub =
        Kokkos::subview( m_hf_v->d_contents1D, std::make_pair( i * mxsz, ( i * mxsz ) + h_szs[i] ) );
    auto b_sub =
        Kokkos::subview( m_hf_v->d_borders1D, std::make_pair( i * mxsz, ( i * mxsz ) + h_szs[i] ) );

    Kokkos::deep_copy( c_sub, ct );
    Kokkos::deep_copy( b_sub, bd );
  }

  hf_ptr->low_edge     = m_hf_v->low_edge.data();
  hf_ptr->h_szs        = m_hf_v->h_szs.data();
  hf_ptr->d_contents1D = m_hf_v->d_contents1D.data();
  hf_ptr->d_borders1D  = m_hf_v->d_borders1D.data();

  m_hf_h = hf_ptr;

  m_hf_dv = Kokkos::View<FHs>( "hfs" );                  // on device
  Kokkos::View<FHs, Kokkos::HostSpace> hfv( m_hf_h );    // wrap host ptr
  Kokkos::deep_copy( m_hf_dv, hfv );                     // copy to device
  m_hf_d = m_hf_dv.data();                               // ptr on device

  Kokkos::View<FHs_v, Kokkos::HostSpace> hv( m_hf_v );
  m_hfv_v = Kokkos::View<FHs_v>( "FHs_v on device" );
  Kokkos::deep_copy(m_hfv_v, hv);
  m_hf_v_d = m_hfv_v.data();


  std::cout << "+++ LD() \n";
  FHs_v ff = *m_hf_v;
  Kokkos::parallel_for ( 1 , KOKKOS_LAMBDA(int bin) {
     printf("%d %d\n",bin,ff.nhist());
     }
  );
  Kokkos::fence();


}
