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
   float* x_ptr ; //[nhits] gpu --values for fill fro each hit
   short * i_ptr ; //[nhits] gpu bin index for each hit
   int * hb_ptr ; //[nbins*1024-]gpu for staged block result
   bool * match ;  // [nhits] GPU, is_matched for each hit
   double * sumw2_array_g ; //[nbins] gpu accumulating each event
   double * ct_array_g ;  // [nbins] gpu accumulating main hist content array
   
   //int * ct_array ; 
   double * ct_array_h ; //cpu final
   double * sumw2_array_h ; //cpu  final sumw2_array

   double sumw_h ;  //host 
   double  sumx_h;   // host final copy out of sumx
   double  sumx2_h;   // host final copy out of sumx2
   double  sumw2_h;   // host final copy out of sumw2
   

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
