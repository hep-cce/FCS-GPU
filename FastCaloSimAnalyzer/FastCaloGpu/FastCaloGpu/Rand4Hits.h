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
     Rand4Hits(){ m_rand_ptr =0 ;
	  }; 
     ~Rand4Hits() {gpuQ(cudaFree(m_rand_ptr));
		 CURAND_CALL(curandDestroyGenerator(m_gen));};
     float *  HitsRandGen(unsigned int nhits, unsigned long long seed ) ;

     float * rand_ptr(){ return m_rand_ptr; };
     void set_rand_ptr( float* ptr) { m_rand_ptr=ptr ; };
     void set_gen( curandGenerator_t  gen) { m_gen=gen ; };
     curandGenerator_t gen() {return m_gen ; } ; 
     void allocate_hist( long long maxhits, unsigned short maxbins,unsigned short maxhitct, int n_hist, int n_match );

     float ** get_F_ptrs(){return m_F_ptrs ; } ;
     double ** get_D_ptrs(){return m_D_ptrs ; } ;
     short ** get_S_ptrs(){return m_S_ptrs ;} ;
     int ** get_I_ptrs(){return m_I_ptrs ;} ;
     bool  ** get_B_ptrs(){return m_B_ptrs ; } ; 
     unsigned long  ** get_Ul_ptrs(){return m_Ul_ptrs ;} ; 
     unsigned long long  ** get_Ull_ptrs(){return m_Ull_ptrs ;} ; 
     unsigned int  ** get_Ui_ptrs(){return m_Ui_ptrs ; } ;
     unsigned long * get_hitcells() { return  m_hitcells ;} ;
     int * get_hitcells_ct() { return  m_hitcells_ct ;} ;

     double ** get_array_h_ptrs() { return m_array_h_ptrs ; } ;
     double ** get_sumw2_array_h_ptrs() { return m_sumw2_array_h_ptrs ; } ;
     double * get_hist_stat_h() {return m_hist_stat_h ; }; 


  private:
      float * m_rand_ptr  ;
      curandGenerator_t m_gen;
      
//patch in some GPU pointers for cudaMalloc
      float ** m_F_ptrs ;
      short ** m_S_ptrs ;
      int ** m_I_ptrs ;
      bool ** m_B_ptrs ;
      unsigned long ** m_Ul_ptrs ; 
      unsigned int ** m_Ui_ptrs ; 
      unsigned long long ** m_Ull_ptrs ; 
      double ** m_D_ptrs;

//host side ;
      unsigned long * m_hitcells;
      int * m_hitcells_ct ;

      double * m_hist_stat_h ;
      double** m_array_h_ptrs ;
      double** m_sumw2_array_h_ptrs ;

};

#endif

