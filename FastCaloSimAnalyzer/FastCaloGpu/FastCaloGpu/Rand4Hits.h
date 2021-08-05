/*
  Copyright (C) 2002-2021 CERN for the benefit of the ATLAS collaboration
*/

#ifndef RAND4HITS_H
#define RAND4HITS_H

#include <stdio.h>
#include <curand.h>

#include "GpuParams.h"
#include "gpuQ.h"
#include "GpuGeneral_structs.h"

#define CURAND_CALL( x )                                                                                               \
  if ( ( x ) != CURAND_STATUS_SUCCESS ) {                                                                              \
    printf("Error at %s:%d\n",__FILE__,__LINE__);\
    exit( EXIT_FAILURE );                                                                                              \
  }

class Rand4Hits {
  public:
  Rand4Hits() {
    m_rand_ptr     = 0;
		m_total_a_hits=0 ;
	  }; 
  ~Rand4Hits() {
    gpuQ( cudaFree( m_rand_ptr ) );
		 CURAND_CALL(curandDestroyGenerator(m_gen));
		cudaFree(m_cells_energy) ;
		cudaFree(m_cell_e) ;
		cudaFree(m_ct) ;
		cudaFree(m_hitparams) ;
		cudaFree(m_simbins) ;
		};
		
     //float *  HitsRandGen(unsigned int nhits, unsigned long long seed ) ;

     float * rand_ptr(int nhits){
	 if( over_alloc(nhits)) {
		 rd_regen() ; 
		return m_rand_ptr ;
	} else {
	  float * f_ptr=&(m_rand_ptr[ 3 * m_current_hits]) ;
	  return f_ptr ; 
	}
	};
     float * rand_ptr_base() {return m_rand_ptr ; }
     void set_rand_ptr( float* ptr) { m_rand_ptr=ptr ; };
     void set_t_a_hits( int nhits) { m_total_a_hits=nhits ; };
     void set_c_hits( int nhits) { m_current_hits=nhits ; };
     unsigned int  get_c_hits( ) { return  m_current_hits ; };
     unsigned int  get_t_a_hits( ) { return  m_total_a_hits ; };
     void set_gen( curandGenerator_t  gen) { m_gen=gen ; };
     curandGenerator_t gen() {return m_gen ; } ; 
  void allocate_hist( long long maxhits, unsigned short maxbins, unsigned short maxhitct, int n_hist, int n_match,
                      bool hitspy );

     void allocate_simulation(  int  maxbins, int maxhitcts,  unsigned long n_cells);

     float * get_cells_energy(){return m_cells_energy ; } ;
     Cell_E * get_cell_e(){return m_cell_e ; } ;
     Cell_E * get_cell_e_h(){return m_cell_e_h ; } ;
     int * get_ct() {return m_ct ; } ;
     int * get_ct_h() {return m_ct_h ; } ;

     HitParams * get_hitparams() {return m_hitparams ; };
     long * get_simbins() {return m_simbins ; } ;

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

     void    rd_regen() {CURAND_CALL(curandGenerateUniform(m_gen, m_rand_ptr, 3*m_total_a_hits) ); }  ;
  void add_a_hits( int nhits ) {
    if ( over_alloc( nhits ) )
      m_current_hits = nhits;
    else
      m_current_hits += nhits;
  };
  bool over_alloc( int nhits ) {
    return m_current_hits + nhits > m_total_a_hits;
  }; // return true if hits over spill, need regenerat rand..

  private:
      float * m_rand_ptr  ;
      unsigned int  m_total_a_hits ;
      unsigned int  m_current_hits ;
      curandGenerator_t m_gen;
      
//patch in some GPU pointers for cudaMalloc
      float * m_cells_energy ;
      Cell_E * m_cell_e ;
      int * m_ct ;

      HitParams * m_hitparams ;
      long * m_simbins ;

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
      Cell_E * m_cell_e_h ;
      int * m_ct_h ;

      double * m_hist_stat_h ;
      double** m_array_h_ptrs ;
      double** m_sumw2_array_h_ptrs ;

};

#endif

