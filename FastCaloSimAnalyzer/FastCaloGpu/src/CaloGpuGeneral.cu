#include "CaloGpuGeneral.h"
#include "GeoRegion.cu"
#include "Hit.h"

#include "gpuQ.h"
#include "Args.h"

#define BLOCK_SIZE 512 
#define NLOOPS 4

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
     float * r= rd4h->HitsRandGen(34, 1234ULL) ;
     test_rand <<< 10, 10 >>> (r ) ;
  cudaDeviceSynchronize() ;
 err = cudaGetLastError();
if (err != cudaSuccess) {
        std::cout<<" testRandom "<< cudaGetErrorString(err)<< std::endl;
     delete rd4h  ;

}

}


__device__ void  rnd_to_fct2d(float& valuex,float& valuey,float rnd0,float rnd1, FH2D* hf2d) {

 int nbinsx=(*hf2d).nbinsx;
 int nbinsy=(*hf2d).nbinsy;
 float * HistoContents= (*hf2d).h_contents ;
 float* HistoBorders= (*hf2d).h_bordersx ;
 float* HistoBordersy= (*hf2d).h_bordersy ; 

 int ibin = nbinsx*nbinsy-1 ;
 for ( int i=0 ; i < nbinsx*nbinsy ; ++i) {
    if   (HistoContents[i]> rnd0 ) {
	 ibin = i ;
	 break ;
	}
 } 


  int biny = ibin/nbinsx;
  int binx = ibin - nbinsx*biny;

  float basecont=0;
  if(ibin>0) basecont=HistoContents[ibin-1];

  float dcont=HistoContents[ibin]-basecont;
  if(dcont>0) {
    valuex = HistoBorders[binx] + (HistoBorders[binx+1]-HistoBorders[binx]) * (rnd0-basecont) / dcont;
  } else {
    valuex = HistoBorders[binx] + (HistoBorders[binx+1]-HistoBorders[binx]) / 2;
  }
  valuey = HistoBordersy[biny] + (HistoBordersy[biny+1]-HistoBordersy[biny]) * rnd1;

}


__device__  float  rnd_to_fct1d( float  rnd, uint32_t* contents, float* borders , int nbins, uint32_t s_MaxValue  ) {


  uint32_t int_rnd=s_MaxValue*rnd;

  int  ibin=nbins-1 ;
  for ( int i=0 ; i < nbins ; ++i) {
    if   (contents[i]> int_rnd ) {
         ibin = i ;
         break ;
        }
  }

  int binx = ibin;

  uint32_t basecont=0;
  if(ibin>0) basecont=contents[ibin-1];

  uint32_t dcont=contents[ibin]-basecont;
  if(dcont>0) {
    return borders[binx] + ((borders[binx+1]-borders[binx]) * (int_rnd-basecont)) / dcont;
  } else {
    return borders[binx] + (borders[binx+1]-borders[binx]) / 2;
  }

}



__device__  void CenterPositionCalculation_d(Hit& hit, const Chain0_Args args) {

    hit.setCenter_r((1.- args.extrapWeight)*args.extrapol_r_ent + 
	args.extrapWeight*args.extrapol_r_ext) ;
    hit.setCenter_z((1.- args.extrapWeight)*args.extrapol_z_ent + 
	args.extrapWeight*args.extrapol_z_ext) ;
    hit.setCenter_eta((1.- args.extrapWeight)*args.extrapol_eta_ent + 
	args.extrapWeight*args.extrapol_eta_ext) ;
    hit.setCenter_phi((1.- args.extrapWeight)*args.extrapol_phi_ent + 
	args.extrapWeight*args.extrapol_phi_ext) ;
}


__device__ void ValidationHitSpy_d( Hit& hit, const  Chain0_Args args ) {


}

__device__ void HistoLateralShapeParametrization_d( Hit& hit, unsigned long t, Chain0_Args args ) {

  int     pdgId    = args.pdgId;
  float  charge   = args.charge;

  int cs=args.charge;
  float center_eta = hit.center_eta();
  float center_phi = hit.center_phi();
  float center_r   = hit.center_r();
  float center_z   = hit.center_z();


  float alpha, r, rnd1, rnd2;
  rnd1 = args.rand[t];
  rnd2 = args.rand[t+args.nhits];

  if(args.is_phi_symmetric) {
    if(rnd2>=0.5) { //Fill negative phi half of shape
      rnd2-=0.5;
      rnd2*=2;
      rnd_to_fct2d(alpha,r,rnd1,rnd2,args.fh2d);
      alpha=-alpha;
    } else { //Fill positive phi half of shape
      rnd2*=2;
      rnd_to_fct2d(alpha,r,rnd1,rnd2,args.fh2d);
    }
  } else {
    rnd_to_fct2d(alpha,r,rnd1,rnd2,args.fh2d);
  }


  float delta_eta_mm = r * cos(alpha);
  float delta_phi_mm = r * sin(alpha);

  // Particles with negative eta are expected to have the same shape as those with positive eta after transformation: delta_eta --> -delta_eta
  if(center_eta<0.)delta_eta_mm = -delta_eta_mm;
  // Particle with negative charge are expected to have the same shape as positively charged particles after transformation: delta_phi --> -delta_phi
  if(charge < 0.)  delta_phi_mm = -delta_phi_mm;

  float dist000    = sqrt(center_r * center_r + center_z * center_z);
  float eta_jakobi = abs(2.0 * exp(-center_eta) / (1.0 + exp(-2 * center_eta)));

  float delta_eta = delta_eta_mm / eta_jakobi / dist000;
  float delta_phi = delta_phi_mm / center_r;

  hit.setEtaPhiZE(center_eta + delta_eta,center_phi + delta_phi,center_z, hit.E());


}

__device__ void HitCellMapping_d( Hit& hit,unsigned long t, Chain0_Args args ) {

 long long  cellele= getDDE(args.geo, args.cs,hit.eta(),hit.phi());

 if( cellele < 0) printf("cellele not found %ld \n", cellele ) ; 

  args.hitcells_b[cellele]= true ;
  args.hitcells[t]=cellele ;
  
/*
  CaloDetDescrElement cell =( *(args.geo)).cells[cellele] ;
  long long id = cell.identify();
  float eta=cell.eta(); 
  float phi=cell.phi();
  float z=cell.z();
  float r=cell.r() ;
*/
}

__device__ void HitCellMappingWiggle_d( Hit& hit,  Chain0_Args args, unsigned long  t ) {

 int nhist=(*(args.fhs)).nhist;
 float*  bin_low_edge = (*(args.fhs)).low_edge ;
 

 float eta =fabs( hit.eta()); 
 if(eta<bin_low_edge[0] || eta> bin_low_edge[nhist]) {
   HitCellMapping_d(hit, t, args) ;

 }

 int bin= nhist ;
  for (int i =0; i< nhist+1 ; ++i ) {
 	if(bin_low_edge[i] > eta ) {
	  bin = i ;
	  break ;
	}
  }

  bin -= 1; 

  uint32_t * contents = (*(args.fhs)).h_contents[bin] ;
  float* borders = (*(args.fhs)).h_borders[bin] ;
  int h_size=(*(args.fhs)).h_szs[bin] ;
  uint32_t s_MaxValue =(*(args.fhs)).s_MaxValue ;
  

     float rnd= args.rand[t+2*args.nhits];

    float wiggle=rnd_to_fct1d(rnd,contents, borders, h_size, s_MaxValue);

    float hit_phi_shifted=hit.phi()+wiggle;
    hit.phi()=Phi_mpi_pi(hit_phi_shifted);
  

  HitCellMapping_d(hit, t,  args) ;

}





__global__  void simulate_chain0_A( float E, int nhits,  Chain0_Args args ) {

  int tid = threadIdx.x + blockIdx.x*blockDim.x ;
  for ( int i=0 ; i<NLOOPS ; ++i ) { 
    unsigned long t = tid+i*gridDim.x*blockDim.x ;
    if ( t  >= nhits ) break ; 
    Hit hit ;
    hit.E()=E ;
    CenterPositionCalculation_d( hit, args) ;
    ValidationHitSpy_d( hit, args) ;
    HistoLateralShapeParametrization_d(hit,t,  args) ;
    HitCellMappingWiggle_d ( hit, args, t ) ;
    ValidationHitSpy_d(hit,args);
//  do something 
//if(t==0) printf("rand(0)=%f\n", args.rand[0]);
  }
 
}




__global__  void simulate_chain0_B1( Chain0_Args args) {

//	printf("From kernel simulate_chain0_B:\n" );
 unsigned long tid = threadIdx.x + blockIdx.x*blockDim.x ;
 if( tid < args.ncells) {
        if(args.hitcells_b[tid]) {
		unsigned int ct = atomicAdd(args.hitcells_ct,1) ;
		args.hitcells_l[ct]=tid; 
//		printf("atomic tid=%lu,ct=%d\n", tid,ct);
	}
 }
}



__global__  void simulate_chain0_C() {

	printf("From kernel simulate_chain0_C:\n" );
}


__host__  void *  CaloGpuGeneral::Rand4Hits_init( long long maxhits, unsigned long long seed ){ 

      Rand4Hits * rd4h = new Rand4Hits ;
	float * f  ;
	curandGenerator_t gen ;
       gpuQ(cudaMalloc((void**)&f , 3*maxhits*sizeof(float))) ;
        
        CURAND_CALL(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));
        CURAND_CALL(curandSetPseudoRandomGeneratorSeed(gen, seed)) ;

         rd4h->set_rand_ptr(f) ;
	 rd4h->set_gen(gen) ;

	return  (void* ) rd4h ;

}

__host__  void   CaloGpuGeneral::Rand4Hits_finish( void * rd4h ){ 

 if ( (Rand4Hits *)rd4h ) delete (Rand4Hits *)rd4h  ;

}


__global__  void simulate_chain0_clean(Chain0_Args args) {
 unsigned long  tid = threadIdx.x + blockIdx.x*blockDim.x ;
 if(tid < args.ncells ) args.hitcells_b[tid]= false ;
 if(tid < args.nhits) args.hitcells_l[tid] = 0 ;
 if(tid ==0 ) args.hitcells_ct[0]= 0 ; 
}


__global__ void simulate_chain0_C1(unsigned int * hitct_b, unsigned int ct , Chain0_Args args) {
extern __shared__ unsigned long hitcells[] ;
unsigned int *  counts = (unsigned int * ) (& hitcells[ct]) ;
 unsigned long  tid = threadIdx.x + blockIdx.x*blockDim.x ;
 unsigned long  hitcell_index=args.hitcells[tid];   //cell index for tid's hit

//read in Hitcells indexes    to shared memeory
for(int j =0 ; j< (ct+blockDim.x -1)/blockDim.x ; ++j ) {
	int index=threadIdx.x+j*blockDim.x;
	if(index < ct ){ 
		hitcells[index]=args.hitcells_l[index] ;
		counts[index]=0 ;
	}
	__syncthreads() ;
}

// save to counts[]
//int iii=0 ;
if(tid < args.nhits) {
  for(unsigned int i=0; i<ct ; ++i) {
//if(tid==0) printf(" i=%d  hitindex=%lu, hitcells=%lu\n", i,hitcell_index, hitcells[i]) ; 
    if(hitcell_index == hitcells[i]) {
      atomicAdd(&(counts[i]), 1) ;
      break ;
    }
//iii++;
  }
}

//	__syncthreads() ;
//if(tid==0) printf("index=%d, count=%d\n", iii, counts[iii] );
//if(tid==0) for( int ii=0 ; ii<ct ; ii++) {printf("from C1 counts[%d]=%d\n" ,ii, counts[ii] ) ;}

__syncthreads() ;
for(int j =0 ; j< (ct+blockDim.x -1)/blockDim.x ; ++j ) {
	int index=threadIdx.x+j*blockDim.x;
	if(index < ct ) 
	hitct_b[ct* blockIdx.x+index]=counts[index] ;
	__syncthreads() ;
}


}

__device__ void warpReduce(volatile int* sdata, int tid) {
sdata[tid] += sdata[tid + 32];
sdata[tid] += sdata[tid + 16];
sdata[tid] += sdata[tid + 8];
sdata[tid] += sdata[tid + 4];
sdata[tid] += sdata[tid + 2];
sdata[tid] += sdata[tid + 1];
}

__global__ void simulate_chain0_D1(unsigned int * hitct_b, int ct_blks, unsigned int ct , Chain0_Args args) {
extern __shared__ int sdata[] ;
int tid=threadIdx.x;

if((tid+blockDim.x) < ct_blks ){
sdata[tid]=hitct_b[tid*ct+blockIdx.x]+hitct_b[(tid+blockDim.x)*ct+blockIdx.x] ;
}else{
sdata[tid]=hitct_b[tid*ct+blockIdx.x] ;
}
__syncthreads();


for (unsigned int s=blockDim.x/2; s>32; s>>=1) {
if (tid < s)
sdata[tid] += sdata[tid + s];
__syncthreads();
}
if (tid < 32) warpReduce(sdata, tid);
if(tid==0)  hitct_b[blockIdx.x] =sdata[0] ;

}



__host__ int highestPowerof2(unsigned int n) 
{ 
    // Invalid input 
    if (n < 1) 
        return 0; 
  
    int res = 1; 
  
    // Try all powers starting from 2^1 
    for (int i=0; i<8*sizeof(unsigned int); i++) 
    { 
        int curr = 1 << i; 
  
        // If current power is more than n, break 
        if (curr > n) 
           break; 
  
        res = curr; 
    } 
  
    return res; 
}

__host__ void CaloGpuGeneral::simulate_hits(float E, int nhits, Chain0_Args& args ) {

        Rand4Hits * rd4h = (Rand4Hits *) args.rd4h ;
	
	
        float * r= rd4h->rand_ptr()  ;
         args.rand = r ;
  CURAND_CALL(curandGenerateUniform(rd4h->gen(), r, 3*nhits));


	 
	unsigned long  ncells = args.ncells ; 
	gpuQ(cudaMalloc((void**)&(args.hitcells_b), args.ncells*sizeof(bool)));
	gpuQ(cudaMalloc((void**)&(args.hitcells_l), args.nhits*sizeof(unsigned long)));
	gpuQ(cudaMalloc((void**)&(args.hitcells), args.nhits*sizeof(unsigned long)));
	gpuQ(cudaMalloc((void**)&(args.hitcells_ct), sizeof(unsigned int)));

        
	int blocksize=BLOCK_SIZE ;
	int threads_tot= args.ncells  ;
	int nblocks= (threads_tot + blocksize-1 )/blocksize ;        

	simulate_chain0_clean <<< nblocks, blocksize >>>( args) ;
  //	cudaDeviceSynchronize() ;
 	cudaError_t err = cudaGetLastError();
 if (err != cudaSuccess) {
        std::cout<< "simulate_chain0_clean "<<cudaGetErrorString(err)<< std::endl;
}


	blocksize=BLOCK_SIZE ;
	threads_tot= (nhits +NLOOPS-1) /NLOOPS  ;
	nblocks= (threads_tot + blocksize-1 )/blocksize ;        

//	 std::cout<<"Nblocks: "<< nblocks << ", blocksize: "<< blocksize 
//                << ", total Threads: " << threads_tot << std::endl ;

  simulate_chain0_A <<<nblocks, blocksize  >>> (E, nhits, args  ) ; 
//  cudaDeviceSynchronize() ;
  err = cudaGetLastError();
 if (err != cudaSuccess) {
        std::cout<< "simulate_chain0_A "<<cudaGetErrorString(err)<< std::endl;
}


  nblocks = (ncells + blocksize -1 )/blocksize ;
  simulate_chain0_B1 <<<nblocks,blocksize >>> (args) ;
//  cudaDeviceSynchronize() ;
 err = cudaGetLastError();
 if (err != cudaSuccess) {
        std::cout<< "simulate_chain0_B1 "<<cudaGetErrorString(err)<< std::endl;

}

   unsigned int ct ;
   gpuQ(cudaMemcpy(&ct, args.hitcells_ct,sizeof(unsigned int), cudaMemcpyDeviceToHost));
   unsigned long *hitcells ;
   hitcells=(unsigned long * ) malloc(sizeof(unsigned long)*ct) ;
   int * hitcells_ct=(int * ) malloc(sizeof(int* )*ct) ;
 // check result 
   gpuQ(cudaMemcpy(hitcells, args.hitcells_l, ct*sizeof(unsigned long), cudaMemcpyDeviceToHost));

//	std::cout<<"hit cell ct="<<ct<<std::endl;
//	std::cout<<"hit cell [0]="<<hitcells[0]<<std::endl;

   blocksize=64 ; 
   nblocks = (args.nhits + blocksize-1)/blocksize ;
	unsigned int * hitcounts_b ;
	gpuQ(cudaMalloc((void**)&(hitcounts_b), ct*nblocks*sizeof(unsigned int)));

    std::cout<<"nblocks for hit counts="<<nblocks<< ", blocksize="<<blocksize<<std::endl ;

   simulate_chain0_C1<<<nblocks, blocksize,ct*(sizeof(unsigned long)+sizeof(unsigned int))>>> (hitcounts_b,ct,args) ;
//  cudaDeviceSynchronize() ;
 err = cudaGetLastError();
 if (err != cudaSuccess) {
        std::cout<< "simulate_chain0_C1 "<<cudaGetErrorString(err)<< std::endl;
}
   

    int ct_b = nblocks ;
    blocksize=highestPowerof2(ct_b);
    nblocks=ct;

    simulate_chain0_D1<<<nblocks,blocksize,blocksize*sizeof(unsigned int) >>>(hitcounts_b,ct_b,ct,args) ;
//  cudaDeviceSynchronize() ;
 err = cudaGetLastError();
 if (err != cudaSuccess) {
        std::cout<< "simulate_chain0_D1 "<<cudaGetErrorString(err)<< std::endl;
}
   gpuQ(cudaMemcpy(hitcells_ct, hitcounts_b, ct*sizeof(int), cudaMemcpyDeviceToHost));

// pass result back 
   args.ct=ct;
   args.hitcells_h=hitcells ;
   args.hitcells_ct_h=hitcells_ct ;

/*int total_ct=0 ;
for(int ii =0 ; ii<ct ; ++ii) {
	std::cout<< "CT["<<hitcells[ii]<<"]=" << hitcells_ct[ii]<<std::endl ;
	total_ct += hitcells_ct[ii] ;
}
std::cout << "Total Counts =" << total_ct << " nhits=" << args.nhits<< std::endl;

  simulate_chain0_C <<< 1,1 >>> () ;
  cudaDeviceSynchronize() ;
 err = cudaGetLastError();
 if (err != cudaSuccess) {
        std::cout<< "simulate_chain0_C "<<cudaGetErrorString(err)<< std::endl;

}
*/


	cudaFree( args.hitcells);
	cudaFree( args.hitcells_ct);
	cudaFree( args.hitcells_b);
	cudaFree( args.hitcells_l);
	cudaFree( hitcounts_b);
//	free(hitcells)  ;
//	free(hitcells_ct)  ;

}




