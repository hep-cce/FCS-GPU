#ifndef LOADGPUFUNCHIST_H
#define LOADGPUFUNCHIST_H

class LoadGpuFuncHist {

struct HistFuncsStr {
	uint32_t s_MaxValue ;
	float * low_edge ;
	unsigned int nhist;
	unsigned int * h_szs ;
	uint32_t * * h_contents ;
	float * * h_borders ;
	} 
	

public :
	__HOST__ LoadFuncHist(){ m_hf=0 ; m_d_hf=0 ; } ;
	__HOST__ ~LoadFuncHist(){ free(m_hf); cuFree(m_d_hf) ; 
			cuFree((*m_d_hf).low_edge);
			cuFree((*m_d_hf).h_szs; 
			for(int i=0 ; i< (*m_d_hf).nhist ; ++i ) 
			 {cuFree((*m_d_hf).h_contents[i]);cuFree((*m_d_hf).h_borders[i]);};
		  } ;


	__HOST__ void set_hf( struct HistFuncsStr * hf_ptr) { m_hf=hf_ptr ; }
	__HOST__ void set_d_hf( struct HistFuncsStr * hf_ptr) { m_d_hf=hf_ptr ; }
	__HOST__ struct HistFuncsStr * hf() {return m_hf ; } ;
	__HOST__ struct HistFuncsStr * d_hf() {return m_d_hf ; } ;

	__HOST__ void LD();

private : 

	 struct HistFuncsStr *  m_hf ;
	 struct HistFuncsStr *  m_d_hf ;

};

#endif


