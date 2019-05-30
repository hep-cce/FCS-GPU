#include "CaloGpuGeneral.h"
#include "GeoRegion.cu"
#include "Hit.h"

#include "gpuQ.h"
#include "Args.h"

#define BLOCK_SIZE 1024 
#define NLOOPS 2

__device__  long long getDDE( GeoGpu* geo, int sampling, float eta, float phi) {

   float * distance = 0 ;
   int * steps =0 ;

int MAX_SAMPLING = geo->max_sample ;
Rg_Sample_Index * SampleIdx = geo->sample_index ;
 GeoRegion * regions_g = geo->regions ;

if(sampling<0) return -1;
  if(sampling>=MAX_SAMPLING) return -1;

   int sample_size= SampleIdx[sampling].size ;
   int sample_index=SampleIdx[sampling].index ;

   GeoRegion * gr = ( GeoRegion *) regions_g ; 
   printf("gr * is %d\n", gr) ; 
  if(sample_size==0) return -1;
  float dist;
  long long bestDDE=-1;
  if(!distance) distance=&dist;
  *distance=+10000000;
  int intsteps;
  int beststeps;
  if(steps) beststeps=(*steps);
   else beststeps=0;

  
  if(sampling<21) {
    for(int skip_range_check=0;skip_range_check<=1;++skip_range_check) {
      for(unsigned int j= sample_index; j< sample_index+sample_size ; ++j) {
        if(!skip_range_check) {
          if(eta< gr[j].mineta()) continue;
          if(eta> gr[j].maxeta()) continue;
        }
    if(steps) intsteps=(*steps);
         else 
       intsteps=0;
        float newdist;
        long long  newDDE= gr[j].getDDE(eta,phi,&newdist,&intsteps);
        if(newdist<*distance) {
          bestDDE=newDDE;
          *distance=newdist;
          if(steps) beststeps=intsteps;
          if(newdist<-0.1) break; //stop, we are well within the hit cell
       }
      }
      if(bestDDE>=0) break;
  }
  } else {
                return -3;
  }
  if(steps) *steps=beststeps;

  return bestDDE;
}

__global__  void testHello_xxx() {

printf("Hello..., I am from GPU threadi..... %d\n", threadIdx.x);

};


__global__  void test_getDDE(GeoGpu * geo , int sample, float eta, float phi) {

long long  index=getDDE(geo, sample, eta,phi) ;

printf("From GPU index of the cell with eta=%f, phi=%f is %ld \n", eta, phi, index) ;

}


__global__  void test_rand(float * r ) {
  int  tid=blockIdx.x*blockDim.x+threadIdx.x ;

  printf("Tid%d  Block ID %d, Thread %d r=[%f]\n", 
         tid, blockIdx.x, threadIdx.x, r[tid] ) ; 



}

__host__ void CaloGpuGeneral::Gpu_Chain_Test() {

        std::cout<< " calling testHelloixxx()"<< std::endl;
 testHello_xxx <<<1, 1>>> () ;
    cudaDeviceSynchronize() ;

 cudaError_t err = cudaGetLastError();
 if (err != cudaSuccess) {
        std::cout<<" testHello "<< cudaGetErrorString(err)<< std::endl;

}

 test_getDDE <<<1,1>>> (GeoLoadGpu::Geo_g,  5, -1.5, 1.6 ) ;
    cudaDeviceSynchronize() ;

 err = cudaGetLastError();
 if (err != cudaSuccess) {
        std::cout<< "test_getDEE "<<cudaGetErrorString(err)<< std::endl;

}


// random number test
     std::cout<<"test Random"<<std::endl ;
     Rand4Hits *  rd4h = new Rand4Hits ; 
     float * r= rd4h->HitsRandGen(1000, 1234ULL) ;
     test_rand <<< 10, 10 >>> (r ) ;
  cudaDeviceSynchronize() ;
 err = cudaGetLastError();
if (err != cudaSuccess) {
        std::cout<<" testRandom "<< cudaGetErrorString(err)<< std::endl;
     delete rd4h  ;

}

}

__device__  void CenterPositionCalculation_d(Hit* hit, const Chain0_Args args) {

    hit->setCenter_r((1.- args.extrapWeight)*args.extrapol_r_ent + 
	args.extrapWeight*args.extrapol_r_ext) ;
    hit->setCenter_z((1.- args.extrapWeight)*args.extrapol_z_ent + 
	args.extrapWeight*args.extrapol_z_ext) ;
    hit->setCenter_r((1.- args.extrapWeight)*args.extrapol_eta_ent + 
	args.extrapWeight*args.extrapol_eta_ext) ;
    hit->setCenter_z((1.- args.extrapWeight)*args.extrapol_phi_ent + 
	args.extrapWeight*args.extrapol_phi_ext) ;
}


__device__ void ValidationHitSpy_d( Hit* hit, const  Chain0_Args args ) {


}

__device__ void HistoLateralShapeParametrization_d( Hit* hit, const  Chain0_Args args ) {


}

__device__ void HitCellMappingWiggle_d( Hit* hit, const  Chain0_Args args ) {


}


__global__  void simulate_chain0_A( float E, int nhits,  Chain0_Args args ) {

  int tid = threadIdx.x + blockIdx.x*blockDim.x ;

  for ( int i=0 ; i<NLOOPS ; ++i ) { 
    if ((tid+i*gridDim.x) >= nhits ) break ;  
    Hit* hit  =new Hit() ;
    hit->E()=E ;
    CenterPositionCalculation_d( hit, args) ;
    ValidationHitSpy_d( hit, args) ;
    HistoLateralShapeParametrization_d(hit, args) ;
    HitCellMappingWiggle_d ( hit, args ) ;
    ValidationHitSpy_d(hit,args);
  } 

}




__global__  void simulate_chain0_B() {

	printf("From kernel simulate_chain0_B:\n" );

}
__global__  void simulate_chain0_C() {

	printf("From kernel simulate_chain0_C:\n" );
}


__host__ void CaloGpuGeneral::simulate_hits(float E, int nhits, Chain0_Args args ) {

	Rand4Hits *  rd4h = new Rand4Hits ;
        float * r= rd4h->HitsRandGen(nhits, args.seed) ;


	int blocksize=BLOCK_SIZE ;
	int threads_tot= (nhits +NLOOPS-1) /NLOOPS  ;
  	
	int nblocks= (threads_tot + blocksize-1 )/blocksize ;        

	 std::cout<<"Nblocks: "<< nblocks << ", blocksize: "<< blocksize 
                << ", total Threads: " << threads_tot << std::endl ;
  simulate_chain0_A <<<nblocks, blocksize  >>> (E, nhits, args  ) ; 
  cudaDeviceSynchronize() ;
 cudaError_t err = cudaGetLastError();
 if (err != cudaSuccess) {
        std::cout<< "simulate_chain0_A "<<cudaGetErrorString(err)<< std::endl;

}


  simulate_chain0_B <<<1,1 >>> () ;
  cudaDeviceSynchronize() ;
 err = cudaGetLastError();
 if (err != cudaSuccess) {
        std::cout<< "simulate_chain0_B "<<cudaGetErrorString(err)<< std::endl;

}
  simulate_chain0_C <<< 1,1 >>> () ;
  cudaDeviceSynchronize() ;
 err = cudaGetLastError();
 if (err != cudaSuccess) {
        std::cout<< "simulate_chain0_C "<<cudaGetErrorString(err)<< std::endl;

}


}




