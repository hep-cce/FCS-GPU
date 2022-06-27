#define CATCH_CONFIG_MAIN  // This tells Catch to provide a main() - only do this in one cpp file
#include "catch.hpp"

#include <chrono>
#include <omp.h>
#include <iostream>

#include "../FastCaloGpu/FastCaloGpu/Rand4Hits.h"
#include <curand.h>

#define MAXHITS 200000
#define MAXBINS 1024

/* In this test, RNs are generated on the CPU and copied to the GPU. 
 * New memory is allocated on CPU, onto which the RNs are copied from
 * the GPU using omp_target_memcpy. RNs originally generated on the CPU
 * are compared with those copied from GPU
 */ 
TEST_CASE( "RNs generated on CPU" ) {
    
  int m_default_device = omp_get_default_device();
  int m_initial_device = omp_get_initial_device();
  std::size_t m_offset = 0;

  long seed = 42;
  long long num = 3 * MAXHITS;
  float one_by_num = 1.0/num;
  float epsilon = 1e-5;

  Rand4Hits* rd4h = new Rand4Hits;
  rd4h->create_gen( seed, num, true ); //true = generate on CPU

  //allocate memory on CPU
  float* rn_cpu_cpy{nullptr};
  rn_cpu_cpy = (float *) malloc( num * sizeof( float ) );
  if ( rn_cpu_cpy == NULL ) {
    std::cout << " ERROR: No space left on host." << std::endl;
  }

  //copy RNs from GPU
  if ( omp_target_memcpy( rn_cpu_cpy, rd4h->rand_ptr_base(), num * sizeof( float ), m_offset, m_offset, m_initial_device, m_default_device ) ) {
     std::cout << "ERROR: copy random numbers from gpu to cpu " << std::endl;
  }

  float cpu_rn_mean(0.0f), gpu_rn_mean(0.0f);
  float cpu_rn_var(0.0f),  gpu_rn_var(0.0f);
  //compare
  for ( int i = 0; i < num; i++ ) {

    cpu_rn_mean += rn_cpu_cpy[i];	  
    gpu_rn_mean += rd4h->get_cpu_rand_at(i);	  
    //std::cout << rd4h->get_rand_at(i) << " " << rn_cpu_cpy[i] << std::endl ;
	 
    cpu_rn_var  += rn_cpu_cpy[i]            * rn_cpu_cpy[i];	  
    gpu_rn_var  += rd4h->get_cpu_rand_at(i) * rd4h->get_cpu_rand_at(i);	  
  }
  cpu_rn_mean *= one_by_num;
  gpu_rn_mean *= one_by_num;

  cpu_rn_var  *= one_by_num;
  gpu_rn_var  *= one_by_num;
  
  //std::cout << "means are " << cpu_rn_mean << " " << gpu_rn_mean << std::endl;
  REQUIRE( std::fabs( cpu_rn_mean - gpu_rn_mean ) < epsilon );
  REQUIRE( std::fabs( cpu_rn_var  - gpu_rn_var  ) < epsilon );

}

/* In this test, RNs are generated separately on the CPU and GPU. 
 * The statistics of the two streams of RNs are compared.
 */ 
TEST_CASE( "Compare stats of RNs generated on CPU and GPU" ) {
    
  int m_default_device = omp_get_default_device();
  int m_initial_device = omp_get_initial_device();
  std::size_t m_offset = 0;

  long seed = 42;
  long long num = 3 * MAXHITS;
  float one_by_num = 1.0/num;
  float epsilon = 5e-4;

  Rand4Hits* rd4h_cpu = new Rand4Hits;
  rd4h_cpu->create_gen( seed, num, true ); //true = generate on CPU

  Rand4Hits* rd4h_gpu = new Rand4Hits;
  rd4h_gpu->create_gen( seed, num, false ); //false = generate on GPU
  
  float* rn_cpu_cpy{nullptr};
  rn_cpu_cpy = (float *) malloc( num * sizeof( float ) );
  if ( rn_cpu_cpy == NULL ) {
    std::cout << " ERROR: No space left on host." << std::endl;
  }

  //copy RNs from GPU
  if ( omp_target_memcpy( rn_cpu_cpy, rd4h_gpu->rand_ptr_base(), num * sizeof( float ), m_offset, m_offset, m_initial_device, m_default_device ) ) {
     std::cout << "ERROR: copy random numbers from gpu to cpu " << std::endl;
  }

  float cpu_rn_mean(0.0f), gpu_rn_mean(0.0f);
  float cpu_rn_var(0.0f),  gpu_rn_var(0.0f);
  //compare
  for ( int i = 0; i < num; i++ ) {

    gpu_rn_mean += rn_cpu_cpy[i];	  
    cpu_rn_mean += rd4h_cpu->get_cpu_rand_at(i);	  
	 
    cpu_rn_var  += rn_cpu_cpy[i]            * rn_cpu_cpy[i];	  
    gpu_rn_var  += rd4h_cpu->get_cpu_rand_at(i) * rd4h_cpu->get_cpu_rand_at(i);	  
  }
  cpu_rn_mean *= one_by_num;
  gpu_rn_mean *= one_by_num;

  cpu_rn_var  *= one_by_num;
  gpu_rn_var  *= one_by_num;
  
  //std::cout << "means are " << cpu_rn_mean << " " << gpu_rn_mean << std::endl;
  //std::cout << "vars are "  << cpu_rn_var  << " " << gpu_rn_var  << std::endl;
  REQUIRE( std::fabs( cpu_rn_mean - gpu_rn_mean ) < epsilon );
  REQUIRE( std::fabs( cpu_rn_var  - gpu_rn_var  ) < epsilon );

}

