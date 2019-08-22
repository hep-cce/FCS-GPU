#ifndef HITSPY_HIST_H
#define HITSPY_HIST_H

#ifndef CUDA_HOSTDEV
#ifdef __CUDACC__
#define CUDA_HOSTDEV __host__ __device__ CUDA_HOSTDEV
#else
#define CUDA_HOSTDEV
#endif
#endif

struct hist1 {
   int	nbin ;   //number of bins
   int nentries ;  //number of enetrys 
   float low ;     //min 
   float up ;     //max
   float* x_ptr ; //gpu value for fill
   short * i_ptr ; //gpu bin index
   int * hb_ptr ; //gpu for staged block result
   bool * match ;
   int * ct_array ;
   float sumx ;
   float sumx2 ;

   CUDA_HOSTDEV inline int find_bin(double x ) {
	if(x<low)  return 0 ;
	else if(x>up ) return nbin+1 ;
	else return  1+ int (nbin*(x-low)/(up-low)) ; 
   }  
} ;

struct Hitspy_Hist {
  hist1  hist_hitgeo_dphi ;
  hist1  hist_hitgeo_matchprevious_dphi ;
};
#endif
