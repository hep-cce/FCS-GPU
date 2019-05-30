#ifndef RAND4HITS_H
#define RAND4HITS_H

#include <stdio.h>
#include <curand.h>

#include "gpuQ.h"


#define CURAND_CALL(x)  if((x)!=CURAND_STATUS_SUCCESS) { \
    printf("Error at %s:%d\n",__FILE__,__LINE__);\
    exit(EXIT_FAILURE) ; }

class Rand4Hits {
  public:
     Rand4Hits(){ m_rand_ptr =0 ;  }; 
     ~Rand4Hits() {gpuQ(cudaFree(m_rand_ptr)); CURAND_CALL(curandDestroyGenerator(m_gen));};

     float *  HitsRandGen(unsigned int nhits, unsigned long long seed ) ;

     curandGenerator_t get_gen(){return m_gen ;};
     float * get_rand_ptr(){ return m_rand_ptr; };
     void set_rand_ptr( float* ptr) { m_rand_ptr=ptr ; };

  private:
      float * m_rand_ptr  ;
      curandGenerator_t m_gen;

};

#endif

