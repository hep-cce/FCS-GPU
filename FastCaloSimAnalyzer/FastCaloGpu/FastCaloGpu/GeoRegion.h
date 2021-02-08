#ifndef GeoRegion_H
#define GeoRegion_H

#include "CaloDetDescrElement_g.h"

// #ifndef CUDA_HOSTDEV
// #  ifdef __CUDACC__
// #    define CUDA_HOSTDEV __host__ __device__
// #  else
// #    ifdef USE_KOKKOS
// #      include <Kokkos_Core.hpp>
// #      include <Kokkos_Random.hpp>
// #      define CUDA_HOSTDEV KOKKOS_INLINE_FUNCTION
// #    else
// #      define CUDA_HOSTDEV
// #    endif
// #  endif
// #endif

#ifdef USE_KOKKOS
#  include <Kokkos_Core.hpp>
#  include <Kokkos_Random.hpp>
#  define __HOSTDEV__ KOKKOS_INLINE_FUNCTION
#else
#  ifdef __CUDACC__
#    define __HOSTDEV__ __host__ __device__
#  else
#    define __HOSTDEV__
#  endif
#endif

__HOSTDEV__ double Phi_mpi_pi( double );

class GeoRegion {
public:
  __HOSTDEV__ GeoRegion() {
    m_all_cells                 = 0;
    m_xy_grid_adjustment_factor = 0;
    m_index                     = 0;
    m_cell_grid_eta             = 0;
    m_cell_grid_phi             = 0;
    m_mineta                    = 0;
    m_maxeta                    = 0;
    m_minphi                    = 0;
    m_maxphi                    = 0;
    m_mineta_raw                = 0;
    m_maxeta_raw                = 0;
    m_minphi_raw                = 0;
    m_maxphi_raw                = 0;
    m_mineta_correction         = 0;
    m_maxeta_correction         = 0;
    m_minphi_correction         = 0;
    m_maxphi_correction         = 0;
    m_deta                      = 0;
    m_dphi                      = 0;
    m_eta_correction            = 0;
    m_phi_correction            = 0;
    m_dphi_double               = 0;
    m_deta_double               = 0;
    m_cells                     = 0;
    m_cells_g                   = 0;
  };

  __HOSTDEV__ ~GeoRegion() { free( m_cells ); };

  __HOSTDEV__ void set_all_cells( CaloDetDescrElement* c ) { m_all_cells = c; };
  __HOSTDEV__ void set_xy_grid_adjustment_factor( float f ) { m_xy_grid_adjustment_factor = f; };
  __HOSTDEV__ void set_index( int i ) { m_index = i; };
  __HOSTDEV__ void set_cell_grid_eta( int i ) { m_cell_grid_eta = i; };
  __HOSTDEV__ void set_cell_grid_phi( int i ) { m_cell_grid_phi = i; };
  __HOSTDEV__ void set_mineta( float f ) { m_mineta = f; };
  __HOSTDEV__ void set_maxeta( float f ) { m_maxeta = f; };
  __HOSTDEV__ void set_minphi( float f ) { m_minphi = f; };
  __HOSTDEV__ void set_maxphi( float f ) { m_maxphi = f; };
  __HOSTDEV__ void set_minphi_raw( float f ) { m_minphi_raw = f; };
  __HOSTDEV__ void set_maxphi_raw( float f ) { m_maxphi_raw = f; };
  __HOSTDEV__ void set_mineta_raw( float f ) { m_mineta_raw = f; };
  __HOSTDEV__ void set_maxeta_raw( float f ) { m_maxeta_raw = f; };
  __HOSTDEV__ void set_mineta_correction( float f ) { m_mineta_correction = f; };
  __HOSTDEV__ void set_maxeta_correction( float f ) { m_maxeta_correction = f; };
  __HOSTDEV__ void set_minphi_correction( float f ) { m_minphi_correction = f; };
  __HOSTDEV__ void set_maxphi_correction( float f ) { m_maxphi_correction = f; };
  __HOSTDEV__ void set_eta_correction( float f ) { m_eta_correction = f; };
  __HOSTDEV__ void set_phi_correction( float f ) { m_phi_correction = f; };
  __HOSTDEV__ void set_deta( float f ) { m_deta = f; };
  __HOSTDEV__ void set_dphi( float f ) { m_dphi = f; };
  __HOSTDEV__ void set_deta_double( float f ) { m_deta_double = f; };
  __HOSTDEV__ void set_dphi_double( float f ) { m_dphi_double = f; };
  __HOSTDEV__ void set_cell_grid( long long* cells ) { m_cells = cells; };
  __HOSTDEV__ void set_cell_grid_g( long long* cells ) { m_cells_g = cells; };

  __HOSTDEV__ long long* cell_grid() const { return m_cells; };
  __HOSTDEV__ long long* cell_grid_g() const { return m_cells_g; };
  __HOSTDEV__ int        cell_grid_eta() const { return m_cell_grid_eta; };
  __HOSTDEV__ int        cell_grid_phi() const { return m_cell_grid_phi; };
  __HOSTDEV__ int        index() const { return m_index; };
  __HOSTDEV__ float      mineta_raw() const { return m_mineta_raw; };
  __HOSTDEV__ float      minphi_raw() const { return m_minphi_raw; };
  __HOSTDEV__ CaloDetDescrElement* all_cells() const { return m_all_cells; };
  __HOSTDEV__ float                maxeta() const { return m_maxeta; };
  __HOSTDEV__ float                mineta() const { return m_mineta; };
  __HOSTDEV__ float                maxphi() const { return m_maxphi; };
  __HOSTDEV__ float                minphi() const { return m_minphi; };

  __HOSTDEV__ int raw_eta_position_to_index( float eta_raw ) const {
    return floor( ( eta_raw - m_mineta_raw ) / m_deta_double );
  };
  __HOSTDEV__ int raw_phi_position_to_index( float phi_raw ) const {
    return floor( ( phi_raw - m_minphi_raw ) / m_dphi_double );
  };

  __HOSTDEV__ bool  index_range_adjust( int& ieta, int& iphi ) const;
  __HOSTDEV__ float calculate_distance_eta_phi( const long long DDE, float eta, float phi, float& dist_eta0,
                                                float& dist_phi0 ) const;

  __HOSTDEV__ long long getDDE( float eta, float phi, float* distance = 0, int* steps = 0 );

protected:
  long long*           m_cells;     // my cells array in the region HOST ptr
  long long*           m_cells_g;   // my cells array in the region gpu ptr
  CaloDetDescrElement* m_all_cells; // all cells in GPU, stored in array.

  float m_xy_grid_adjustment_factor;
  int   m_index;
  int   m_cell_grid_eta, m_cell_grid_phi;
  float m_mineta, m_maxeta, m_minphi, m_maxphi;
  float m_mineta_raw, m_maxeta_raw, m_minphi_raw, m_maxphi_raw;
  float m_mineta_correction, m_maxeta_correction, m_minphi_correction, m_maxphi_correction;
  float m_deta, m_dphi, m_eta_correction, m_phi_correction;
  float m_dphi_double, m_deta_double;
};
#endif
