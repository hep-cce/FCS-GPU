#include "Rand4Hits.h"

float *  Rand4Hits::HitsRandGen(unsigned int nhits, unsigned long long seed ) {

  gpuQ(cudaMalloc((void**)&m_rand_ptr , 3*nhits*sizeof(float))) ;
  CURAND_CALL(curandCreateGenerator(&m_gen, 
                CURAND_RNG_PSEUDO_DEFAULT));
  CURAND_CALL(curandSetPseudoRandomGeneratorSeed(m_gen, seed)) ;

  CURAND_CALL(curandGenerateUniform(m_gen, m_rand_ptr, 3*nhits));

   return m_rand_ptr ;
} 


