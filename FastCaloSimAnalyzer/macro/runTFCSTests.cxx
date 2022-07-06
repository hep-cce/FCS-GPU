#include <omp.h>
#include <iostream>
#include <curand.h>

#define CATCH_CONFIG_MAIN  // This tells Catch to provide a main() - only do this in one cpp file
#include "catch.hpp"
#include "../FastCaloGpu/FastCaloGpu/Rand4Hits.h"
#include "../FastCaloGpu/FastCaloGpu/GeoLoadGpu.h"

#include "../FastCaloSimAnalyzer/CaloGeometryFromFile.h"
#include "../FastCaloSimCommon/src/TFCSSampleDiscovery.h"

#define MAXHITS 200000
#define MAXBINS 1024

template <typename T>
void copy_rands_to (T* dest, T* src, const int num) {

  int m_default_device = omp_get_default_device();
  int m_initial_device = omp_get_initial_device();
  std::size_t m_offset = 0;

  if ( omp_target_memcpy( dest, src, num * sizeof( T ), m_offset, m_offset, m_initial_device, m_default_device ) ) {
     std::cout << "ERROR: copy random numbers from gpu to cpu " << std::endl;
  }

}

template <typename T>
std::array<T, 2> calculate_moments (T* rand_stream, const int num) {

  std::array<T, 2> moments = {0.0f, 0.0f};

  for ( int i = 0; i < num; i++ ) {

    moments[0] += rand_stream[i];
    moments[1] += rand_stream[i] * rand_stream[i];
  }

  T one_by_num = 1.0/num;
  moments[0] *= one_by_num;
  moments[1] *= one_by_num;

  return moments;
}

/* In this test, RNs are generated on the CPU and copied to the GPU.
 * New memory is allocated on CPU, onto which the RNs are copied from
 * the GPU using omp_target_memcpy. RNs originally generated on the CPU
 * are compared with those copied from GPU
 */
TEST_CASE( "RNs generated on CPU copied to GPU" ) {

  int m_default_device = omp_get_default_device();

  const long seed = 42;
  const long long num = 3 * MAXHITS;
  const float epsilon = 1e-5;

  Rand4Hits* rd4h = new Rand4Hits;
  rd4h->create_gen( seed, num, true ); //true = generate on CPU

  //allocate memory on CPU
  float* rn_gpu_cpy{nullptr};
  rn_gpu_cpy = (float *) malloc( num * sizeof( float ) );
  if ( rn_gpu_cpy == NULL ) {
    std::cout << " ERROR: No space left on host." << std::endl;
  }

  //copy RNs from GPU
  copy_rands_to ( rn_gpu_cpy, rd4h->rand_ptr_base(), num );

  //compare
  std::array <float, 2> gpu_moments = calculate_moments ( rn_gpu_cpy, num );
  std::array <float, 2> cpu_moments = calculate_moments ( rd4h->rnd_ptr_cpu(), num );

  REQUIRE( std::fabs( cpu_moments[0] - gpu_moments[0] ) < epsilon );
  REQUIRE( std::fabs( cpu_moments[1] - gpu_moments[1] ) < epsilon );

}

/* In this test, RNs are generated separately on the CPU and GPU.
 * The statistics of the two streams of RNs are compared.
 */
TEST_CASE( "Compare stats of RNs generated on CPU and GPU" ) {

  const long seed = 42;
  const long long num = 3 * MAXHITS;
  const float epsilon = 5e-4;

  Rand4Hits* rd4h_cpu = new Rand4Hits;
  rd4h_cpu->create_gen( seed, num, true ); //true = generate on CPU

  Rand4Hits* rd4h_gpu = new Rand4Hits;
  rd4h_gpu->create_gen( seed, num, false ); //false = generate on GPU

  float* rn_gpu_cpy{nullptr};
  rn_gpu_cpy = (float *) malloc( num * sizeof( float ) );
  if ( rn_gpu_cpy == NULL ) {
    std::cout << " ERROR: No space left on host." << std::endl;
  }

  //copy RNs from GPU
  copy_rands_to ( rn_gpu_cpy, rd4h_gpu->rand_ptr_base(), num );

  //compare
  std::array <float, 2> gpu_moments = calculate_moments ( rn_gpu_cpy, num );
  std::array <float, 2> cpu_moments = calculate_moments ( rd4h_cpu->rnd_ptr_cpu(), num );

  REQUIRE( std::fabs( cpu_moments[0] - gpu_moments[0] ) < epsilon );
  REQUIRE( std::fabs( cpu_moments[1] - gpu_moments[1] ) < epsilon );
}




inline void region_data_cpy( CaloGeometryLookup* glkup, GeoRegion* gr ) {

  // Copy all parameters
  gr->set_xy_grid_adjustment_factor( glkup->xy_grid_adjustment_factor() );
  gr->set_index( glkup->index() );

  int neta = glkup->cell_grid_eta();
  int nphi = glkup->cell_grid_phi();
  // std::cout << " copy region " << glkup->index() << "neta= " << neta<< ", nphi= "<<nphi<< std::endl ;

  gr->set_cell_grid_eta( neta );
  gr->set_cell_grid_phi( nphi );

  gr->set_mineta( glkup->mineta() );
  gr->set_minphi( glkup->minphi() );
  gr->set_maxeta( glkup->maxeta() );
  gr->set_maxphi( glkup->maxphi() );

  gr->set_mineta_raw( glkup->mineta_raw() );
  gr->set_minphi_raw( glkup->minphi_raw() );
  gr->set_maxeta_raw( glkup->maxeta_raw() );
  gr->set_maxphi_raw( glkup->maxphi_raw() );

  gr->set_mineta_correction( glkup->mineta_correction() );
  gr->set_minphi_correction( glkup->minphi_correction() );
  gr->set_maxeta_correction( glkup->maxeta_correction() );
  gr->set_maxphi_correction( glkup->maxphi_correction() );

  gr->set_eta_correction( glkup->eta_correction() );
  gr->set_phi_correction( glkup->phi_correction() );
  gr->set_deta( glkup->deta() );
  gr->set_dphi( glkup->dphi() );

  gr->set_deta_double( glkup->deta_double() );
  gr->set_dphi_double( glkup->dphi_double() );

  // now cell array copy from GeoLookup Object
  // new cell_grid is a unsigned long array
  long long* cells = (long long*)malloc( sizeof( long long ) * neta * nphi );
  gr->set_cell_grid( cells );

  if ( neta != (int)( *( glkup->cell_grid() ) ).size() )
    std::cout << "neta " << neta << ", vector eta size " << ( *( glkup->cell_grid() ) ).size() << std::endl;
  for ( int ie = 0; ie < neta; ++ie ) {
    //    	if(nphi != (*(glkup->cell_grid()))[ie].size() )
    //		 std::cout<<"neta " << neta << "nphi "<<nphi <<", vector phi size "<<  (*(glkup->cell_grid()))[ie].size()
    //<< std::endl;

    for ( int ip = 0; ip < nphi; ++ip ) {

      //	if(glkup->index()==0 ) std::cout<<"in loop.."<< ie << " " <<ip << std::endl;
      auto c = ( *( glkup->cell_grid() ) )[ie][ip];
      if ( c ) {
        cells[ie * nphi + ip] = c->calo_hash();

      } else {
        cells[ie * nphi + ip] = -1;
        //	        std::cout<<"NUll cell in loop.."<< ie << " " <<ip << std::endl;
      }
    }
  }
}

inline void GeoLg( CaloGeometryFromFile* m_geo, GeoLoadGpu* m_gl ) {
  m_gl = new GeoLoadGpu();
  m_gl->set_ncells( m_geo->get_cells()->size() );
  m_gl->set_max_sample( CaloGeometry::MAX_SAMPLING );
  int nrgns = m_geo->get_tot_regions();

  std::cout << "Total GeoRegions= " << nrgns << std::endl;
  std::cout << "Total cells= " << m_geo->get_cells()->size() << std::endl;

  m_gl->set_nregions( nrgns );
  m_gl->set_cellmap( m_geo->get_cells() );

  GeoRegion* GR_ptr = (GeoRegion*)malloc( nrgns * sizeof( GeoRegion ) );
  m_gl->set_regions( GR_ptr );

  Rg_Sample_Index* si = (Rg_Sample_Index*)malloc( CaloGeometry::MAX_SAMPLING * sizeof( Rg_Sample_Index ) );

  m_gl->set_sample_index_h( si );

  int i = 0;
  for ( int is = 0; is < CaloGeometry::MAX_SAMPLING; ++is ) {
    si[is].index = i;
    int nr       = m_geo->get_n_regions( is );
    si[is].size  = nr;
    for ( int ir = 0; ir < nr; ++ir ) region_data_cpy( m_geo->get_region( is, ir ), &GR_ptr[i++] );
    //    std::cout<<"Sample " << is << "regions: "<< nr << ", Region Index " << i << std::endl ;
  }
}

/* In this test, Geometry parameters from GPU are copied to CPU and
 * compared with the original on CPU.
 */
TEST_CASE( "Compare Geometry Params" ) {

  CaloGeometryFromFile* m_geo = new CaloGeometryFromFile();

  std::string GeometryPath = "/work/atif/FastCaloSimInputs";

  // load geometry files
  m_geo->LoadGeometryFromFile( GeometryPath + "/CaloGeometry/Geometry-ATLAS-R2-2016-01-00-01.root",
                               "ATLAS-R2-2016-01-00-01",
                               GeometryPath + "/CaloGeometry/cellId_vs_cellHashId_map.txt" );

  std::string geoPathFCal1 = GeometryPath + "/CaloGeometry/FCal1-electrodes.sorted.HV.09Nov2007.dat";
  std::string geoPathFCal2 = GeometryPath + "/CaloGeometry/FCal2-electrodes.sorted.HV.April2011.dat";
  std::string geoPathFCal3 = GeometryPath + "/CaloGeometry/FCal3-electrodes.sorted.HV.09Nov2007.dat";
  m_geo->LoadFCalGeometryFromFiles( TFCSSampleDiscovery::geometryNameFCal() );

  //std::cout << "------ m_geo->get_cells()->size() = " << m_geo->get_cells()->size() << std::endl;
  
  GeoLoadGpu* m_gl;
  GeoLg ( m_geo, m_gl );
  m_gl->LoadGpu();

//  /********** Test ********/
//  GeoRegion* m_regions_host{0};
//  m_regions_host = (GeoRegion *) malloc( sizeof( GeoRegion ) * m_nregions );
//  if ( omp_target_memcpy( m_regions_host, m_regions_d, sizeof( GeoRegion ) * m_nregions,
//                                    m_offset, m_offset, m_initial_device, m_default_device ) ) {
//    std::cout << "ERROR: copy m_regions from device to host." << std::endl;
//    return false;
//  }
//  std::cout << "Comparing GeoRegion members from dev and host" << std::endl;
//  std::cout << m_regions->mineta_raw() << " " << m_regions_host->mineta_raw()  << std::endl;
//  /************************/
//
//
//  GeoLoadGpu* m_gl;
//  if ( m_gl->LoadGpu() ) std::cout << "GPU Geometry loaded!!!" << std::endl;
//
//  /********** Test ********/
//  GeoGpu* Gptr_host;
//  Gptr_host = (GeoGpu *) malloc( sizeof( GeoGpu ) );
//  if ( omp_target_memcpy( Gptr_host, Gptr, sizeof( GeoGpu ),
//               m_offset, m_offset, m_initial_device, m_default_device ) ) {
//    std::cout << "ERROR: copy Gptr from device to host " << std::endl;
//    return false;
//  }
//  std::cout << "Comparing Gptr members from dev and host" << std::endl;
//  std::cout << geo_gpu_h.cells        << " " << Gptr_host->cells        << std::endl;
//  std::cout << geo_gpu_h.ncells       << " " << Gptr_host->ncells       << std::endl;
//  std::cout << geo_gpu_h.nregions     << " " << Gptr_host->nregions     << std::endl;
//  std::cout << geo_gpu_h.regions      << " " << Gptr_host->regions      << std::endl;
//  std::cout << geo_gpu_h.max_sample   << " " << Gptr_host->max_sample   << std::endl;
//  std::cout << geo_gpu_h.sample_index << " " << Gptr_host->sample_index << std::endl;
//  /************************/
//
//
//
//  REQUIRE( std::fabs( cpu_moments[1] - gpu_moments[1] ) < epsilon );
}


