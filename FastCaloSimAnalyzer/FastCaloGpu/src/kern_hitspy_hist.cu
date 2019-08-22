// kernels to calculate Historgarm(counts) 
// after bin number already identified and stored in memeory for each hit



__global__ void hitspy_hist_stgA(short *i, int nbins, int nhits, int* b_result ) {
extern __shared__ unsigned int  counts[] ;
unsigned long tid = threadIdx.x+ blockIdx.x*blockDim.x ;

//zero shared memory 
 for (int j=0; j<(nbins+blockDim.x-1)/blockDim.x ; ++j) {// when nbin > blockDim
  int index = j*blockDim.x+ threadIdx.x ;
  if (index < nbins) counts[index]=0 ;
 }
 __syncthreads() ;


//Atomic update counts
 if(tid < nhits ) {
  int my_bin = i[tid] ;
  atomicAdd(&counts[my_bin], 1) ;
 }
 __syncthreads();

//write out result
 for (int j=0; j<(nbins+blockDim.x-1)/blockDim.x ; ++j) {// when nbin > blockDim
  int index = j*blockDim.x+ threadIdx.x ;
  if (index < nbins) b_result[index+nbins*blockIdx.x] = counts[index] ;
  __syncthreads() ;
 }


}




//This one calculate blockwise atomic counts for matched hits,  save to b_out
__global__ void hitspy_hist_matched_stgA(short *i, bool* match , int nbins, int nhits, int* b_result ) {
extern __shared__ unsigned int  counts[] ;
unsigned long tid = threadIdx.x+ blockIdx.x*blockDim.x ;

//zero shared memory 
 for (int j=0; j<(nbins+blockDim.x-1)/blockDim.x ; ++j) {// when nbin > blockDim
  int index = j*blockDim.x+ threadIdx.x ;
  if (index < nbins) counts[index]=0 ;
 }
 __syncthreads() ;


//Atomic update counts
 if(tid < nhits ) {
  int my_bin = i[tid] ;
  bool is_match = match[tid] ;
  if(is_match) atomicAdd(&counts[my_bin], 1) ;
 }
 __syncthreads();

//write out result
 for (int j=0; j<(nbins+blockDim.x-1)/blockDim.x ; ++j) {// when nbin > blockDim
  int index = j*blockDim.x+ threadIdx.x ;
  if (index < nbins) b_result[index+nbins*blockIdx.x] = counts[index] ;
  __syncthreads() ;
 }

}




//This kernel merge block counts results to final counts 
// ct_blks  is number of blocks need to merge
// assume blockDim.x in 2^n   should be largest 2^n  less or eq ct_blks/2 
__global__ void hitspy_hist_stgB(int* b_results, int ct_blks, int nbins ) {
extern __shared__ int sdata[] ;
int tid = threadIdx.x ;

if((tid+ blockDim.x) < ct_blks ) {
  sdata[tid]= b_results[tid*nbins+blockIdx.x] + b_results[(tid+blockDim.x)*nbins+blockIdx.x] ;
  sdata[tid]= (tid < ct_blks) ? b_results[tid*nbins+blockIdx.x] : 0 ;
}
__syncthreads() ;

for (unsigned int s=blockDim.x/2; s>32; s>>=1 ) {
if(tid <s ) 
 sdata[tid] += sdata[tid + s] ;
__syncthreads() ;
}

if(tid<32) warpReduce(sdata, tid) ;

//now thread 0 has the sum 
if(tid==0) b_results[blockIdx.x]= sdata[0] ; 

}




__device__ void warpReducef(volatile float* sdata, int tid) {
sdata[tid] += sdata[tid + 32];
sdata[tid] += sdata[tid + 16];
sdata[tid] += sdata[tid + 8];
sdata[tid] += sdata[tid + 4];
sdata[tid] += sdata[tid + 2];
sdata[tid] += sdata[tid + 1];
}



// kernel for sum x, x2 
//assume block size  2^n , should be  largest 2^n less than or eq nhits/2 
__global__ void hitspy_hist_sumx_stgA( float * x, int nhits, float *sumx_b, float *sumx2_b  ) {
extern __shared__ float sfdata[] ;

int tid1 = threadIdx.x + 2*blockDim.x* blockIdx.x ;
int tid2 = tid1+blockDim.x ;
int tid = threadIdx.x ;

float x1= (tid1 < nhits) ? x[tid1] : 0  ;
float x2= (tid2 < nhits) ? x[tid2] : 0 ;
sfdata[threadIdx.x]= x1 + x2 ;
sfdata[threadIdx.x+blockDim.x] = x1*x1+ x2*x2 ;

__syncthreads() ; 

for (unsigned int s=blockDim.x/2; s>32;  s>>=1 ) {
if(tid <s ) {
 sfdata[tid] += sfdata[tid + s] ;
 sfdata[tid+blockDim.x]=sfdata[tid+blockDim.x +s ] ;
 }
__syncthreads() ;
}

if(tid<32) {
warpReducef(sfdata, tid) ;
warpReducef(&sfdata[blockDim.x], tid) ;
}

//write result out 
if(tid==0) {
sumx_b[blockIdx.x]=sfdata[0] ;
sumx2_b[blockIdx.x]=sfdata[blockDim.x] ;

}


}

// kernel for sum x, x2 for matched hits 
//assume block size  2^n , should be  largest 2^n less than or eq nhits/2 
__global__ void hitspy_hist_sumx_matched_stgA( float * x, int nhits, float *sumx_b, float *sumx2_b, bool* match ) {
extern __shared__ float sfdata[] ;

int tid1 = threadIdx.x + 2*blockDim.x* blockIdx.x ;
int tid2 = tid1+blockDim.x ;
int tid = threadIdx.x ;

bool m1=match[tid1] ;
bool m2=match[tid2] ;

float x1= (tid1 < nhits && m1 ) ? x[tid1] : 0  ;
float x2= (tid2 < nhits && m2 ) ? x[tid2] : 0 ;
sfdata[threadIdx.x]= x1 + x2 ;
sfdata[threadIdx.x+blockDim.x] = x1*x1+ x2*x2 ;

__syncthreads() ; 

for (unsigned int s=blockDim.x/2; s>32 ;  s>>=1 ) {
 if(tid <s ) { 
  sfdata[tid] += sfdata[tid + s] ;
  sfdata[tid+blockDim.x]=sfdata[tid+blockDim.x +s ];
 }

__syncthreads() ;
}

if(tid<32) {
warpReducef(sfdata, tid) ;
warpReducef(&sfdata[blockDim.x], tid) ;
}

//write result out 
if(tid==0) {
sumx_b[blockIdx.x]=sfdata[0] ;
sumx2_b[blockIdx.x]=sfdata[blockDim.x] ;

}


}
 
// second stage  sum  for x and x^2 
__global__ void hitspy_hist_sumx_stgB( int sum_blks, float *sumx_b, float *sumx2_b  ) {
extern __shared__ float sfdata[] ;

int tid = threadIdx.x ;

if((tid+blockDim.x) < sum_blks ) {
 sfdata[tid]=sumx_b[tid]+sumx_b[tid+blockDim.x] ;
 sfdata[tid+blockDim.x] = sumx2_b[tid]+sumx2_b[tid+blockDim.x] ;
} else {
 sfdata[tid]=(tid < sum_blks) ? sumx_b[tid] : 0 ;
 sfdata[tid+blockDim.x]=(tid < sum_blks) ? sumx2_b[tid] : 0 ;
}

__syncthreads() ;

for (unsigned int s=blockDim.x/2; s>32 ; s>>=1 ) {
 if(tid <s ) {
  sfdata[tid] += sfdata[tid + s] ;
  sfdata[tid+blockDim.x]=sfdata[tid+blockDim.x +s ] ;
 }
__syncthreads() ;
}

if(tid<32) {
warpReducef(sfdata, tid) ;
warpReducef(&sfdata[blockDim.x], tid) ;
}

if(tid==0) {
sumx_b[0]=sfdata[0] ;
sumx2_b[0]=sfdata[blockDim.x] ;
}

} 

