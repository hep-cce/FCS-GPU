/*
  Copyright (C) 2002-2021 CERN for the benefit of the ATLAS collaboration
*/

#ifndef LOADGPUFUNCHIST_H
#define LOADGPUFUNCHIST_H

#include <omp.h>
#include "FH_structs.h"

class LoadGpuFuncHist {

public :
  LoadGpuFuncHist() {
    m_hf     = 0;
    m_d_hf   = 0;
    m_d_hf2d = 0;
    m_hf2d   = 0;
    m_hf_d   = 0;
    m_hf2d_d = 0;
  };
	~LoadGpuFuncHist() ;


	void set_hf( FHs * hf_ptr) { m_hf=hf_ptr ; }
	void set_d_hf( FHs * hf_ptr) { m_d_hf=hf_ptr ; }
	void set_hf2d( FH2D * hf_ptr) { m_hf2d=hf_ptr ; }
	void set_d_hf2d( FH2D * hf_ptr) { m_d_hf2d=hf_ptr ; }
	FHs * hf() {return m_hf ; } ;
	FHs * d_hf() {return m_d_hf ; } ;
	FH2D * hf2d() {return m_hf2d ; } ;
	FH2D * hf2d_d() {return m_hf2d_d ; } ;
	FH2D * d_hf2d() {return m_d_hf2d ; } ;

	void LD();
	void LD2D();

private : 
 	 int m_default_device = omp_get_default_device();
         int m_initial_device = omp_get_initial_device();
         std::size_t m_offset = 0;

	 struct FHs * m_hf ;      
	 struct FHs *  m_d_hf ;  //device pointer
	 struct FHs *  m_hf_d ;  //host pointer to struct hold device param that is copied to device
	 struct FH2D * m_hf2d ;
	 struct FH2D * m_hf2d_d ;  //host poniter struct hold device param to be copied to device
	 struct FH2D * m_d_hf2d ;   //device pointer
};

#endif


