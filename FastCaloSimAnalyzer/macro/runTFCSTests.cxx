#include <omp.h>
#include <iostream>
#include <curand.h>

#define CATCH_CONFIG_MAIN  // This tells Catch to provide a main() - only do this in one cpp file
#include "catch.hpp"
#include "../FastCaloGpu/FastCaloGpu/Rand4Hits.h"
#include "../FastCaloGpu/FastCaloGpu/GeoLoadGpu.h"

//#include "TFCSSampleDiscovery.h"
#include "CaloGeometryFromFile.h"

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

/* In this test, Geometry parameters from GPU are copied to CPU and
 * compared with the original on CPU.
 */
TEST_CASE( "Compare Geometry Params" ) {

  CaloGeometryFromFile* m_geo = new CaloGeometryFromFile();

  std::string GeometryPath = "/work/atif/FastCaloSimInputs/CaloGeometry/";

  // load geometry files
  //m_geo->LoadGeometryFromFile( "/work/atif/FastCaloSimInputs/CaloGeometry/Geometry-ATLAS-R2-2016-01-00-01.root",
        //                     "ATLAS-R2-2016-01-00-01",
          //                     "/work/atif/FastCaloSimInputs/CaloGeometry/cellId_vs_cellHashId_map.txt" );

  //m_geo->LoadFCalGeometryFromFiles( TFCSSampleDiscovery::geometryNameFCal() );

  std::cout << "------ m_geo->get_cells()->size() = " << m_geo->get_cells()->size() << std::endl;
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


