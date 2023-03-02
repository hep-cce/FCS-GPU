/*
  Copyright (C) 2002-2021 CERN for the benefit of the ATLAS collaboration
*/

#ifndef GeoRegion_H
#define GeoRegion_H

#include <cmath>
#include "CaloDetDescrElement_g.h"

#ifndef CUDA_HOSTDEV
#ifdef __CUDACC__
#define CUDA_HOSTDEV __host__ __device__
#else
#define CUDA_HOSTDEV inline
#endif
#endif

#    define __HOSTDEV__ inline
__HOSTDEV__ double Phi_mpi_pi( double );

class GeoRegion {
    public:
        CUDA_HOSTDEV GeoRegion() {
	m_all_cells = 0 ;
	m_xy_grid_adjustment_factor=0 ;
	m_index = 0 ;
	m_cell_grid_eta=0;
	m_cell_grid_phi=0;
	m_mineta=0;
	m_maxeta=0;
	m_minphi=0;
	m_maxphi=0;
	m_mineta_raw=0;
	m_maxeta_raw=0;
	m_minphi_raw=0;
	m_maxphi_raw=0;
	m_mineta_correction=0;
	m_maxeta_correction=0;
	m_minphi_correction=0;
	m_maxphi_correction=0;
	m_deta=0;
	m_dphi=0;
	m_eta_correction=0;
	m_phi_correction=0;
	m_dphi_double=0 ;
	m_deta_double=0 ;
	m_cells=0 ;
	m_cells_g=0;
	} ;

        CUDA_HOSTDEV ~GeoRegion() { free( m_cells) ;} ;


    CUDA_HOSTDEV void set_all_cells(CaloDetDescrElement * c) {m_all_cells = c ; };
    CUDA_HOSTDEV void set_xy_grid_adjustment_factor(float f) {m_xy_grid_adjustment_factor=f ; };
    CUDA_HOSTDEV void set_index(int i ) {m_index=i ; };
    CUDA_HOSTDEV void set_cell_grid_eta(int i) {m_cell_grid_eta=i ; };
    CUDA_HOSTDEV void set_cell_grid_phi(int i) {m_cell_grid_phi=i ; };
    CUDA_HOSTDEV void set_mineta(float f) {m_mineta=f ; };
    CUDA_HOSTDEV void set_maxeta(float f) {m_maxeta=f ; };
    CUDA_HOSTDEV void set_minphi(float f) {m_minphi=f ; };
    CUDA_HOSTDEV void set_maxphi(float f) {m_maxphi=f ; };
    CUDA_HOSTDEV void set_minphi_raw(float f) {m_minphi_raw=f ; };
    CUDA_HOSTDEV void set_maxphi_raw(float f) {m_maxphi_raw=f ; };
    CUDA_HOSTDEV void set_mineta_raw(float f) {m_mineta_raw=f ; };
    CUDA_HOSTDEV void set_maxeta_raw(float f) {m_maxeta_raw=f ; };
    CUDA_HOSTDEV void set_mineta_correction(float f) {m_mineta_correction=f ; };
    CUDA_HOSTDEV void set_maxeta_correction(float f) {m_maxeta_correction=f ; };
    CUDA_HOSTDEV void set_minphi_correction(float f) {m_minphi_correction=f ; };
    CUDA_HOSTDEV void set_maxphi_correction(float f) {m_maxphi_correction=f ; };
    CUDA_HOSTDEV void set_eta_correction(float f) {m_eta_correction=f ; };
    CUDA_HOSTDEV void set_phi_correction(float f) {m_phi_correction=f ; };
    CUDA_HOSTDEV void set_deta(float f) {m_deta=f ; };
    CUDA_HOSTDEV void set_dphi(float f) {m_dphi=f ; };
    CUDA_HOSTDEV void set_deta_double(float f) {m_deta_double=f ; };
    CUDA_HOSTDEV void set_dphi_double(float f) {m_dphi_double=f ; };
    CUDA_HOSTDEV void set_cell_grid( long long * cells ) {m_cells= cells ; };
    CUDA_HOSTDEV void set_cell_grid_g( long long * cells ) {m_cells_g = cells ; };


    CUDA_HOSTDEV long long * cell_grid( ) { return m_cells ; };
    CUDA_HOSTDEV long long * cell_grid_g( ) { return m_cells_g ; };
    CUDA_HOSTDEV int cell_grid_eta( ) { return m_cell_grid_eta ; };
    CUDA_HOSTDEV int cell_grid_phi( ) { return m_cell_grid_phi ; };
    CUDA_HOSTDEV int index() {return m_index ; };
    CUDA_HOSTDEV float mineta_raw( ) {return m_mineta_raw ; };
    CUDA_HOSTDEV float minphi_raw( ) {return m_minphi_raw ; };
    CUDA_HOSTDEV CaloDetDescrElement *all_cells( ) {return m_all_cells ; };
    CUDA_HOSTDEV float maxeta() {return m_maxeta ; };
    CUDA_HOSTDEV float mineta() {return m_mineta ; };
    CUDA_HOSTDEV float maxphi() {return m_maxphi ; };
    CUDA_HOSTDEV float minphi() {return m_minphi ; };

  CUDA_HOSTDEV int raw_eta_position_to_index( float eta_raw ) const {
    return std::floor( ( eta_raw - m_mineta_raw ) / m_deta_double );
  };
  CUDA_HOSTDEV int raw_phi_position_to_index( float phi_raw ) const {
    return std::floor( ( phi_raw - m_minphi_raw ) / m_dphi_double );
  };

    CUDA_HOSTDEV bool index_range_adjust(int& ieta,int& iphi) ;
  CUDA_HOSTDEV float calculate_distance_eta_phi( const long long DDE, float eta, float phi, float& dist_eta0,
                                                 float& dist_phi0 );

    CUDA_HOSTDEV long long  getDDE(float eta,float phi,float* distance=0,int* steps=0) ;


    protected: 


      long long * m_cells  ;  // my cells array in the region HOST ptr
      long long * m_cells_g  ;  // my cells array in the region gpu ptr
    CaloDetDescrElement * m_all_cells ;   // all cells in GPU, stored in array.


    float m_xy_grid_adjustment_factor;
    int m_index ;
    int m_cell_grid_eta,m_cell_grid_phi; 
    float m_mineta,m_maxeta,m_minphi,m_maxphi;
    float m_mineta_raw,m_maxeta_raw,m_minphi_raw,m_maxphi_raw;
    float m_mineta_correction,m_maxeta_correction,m_minphi_correction,m_maxphi_correction;
    float  m_deta,m_dphi,m_eta_correction,m_phi_correction;
    float  m_dphi_double, m_deta_double ;

};
#endif
