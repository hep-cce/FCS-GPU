#include "CaloGpuGeneral.h"
#include "GeoRegion.cu"
#include "Hit.h"

#include "gpuQ.h"
#include "Args.h"

#define BLOCK_SIZE 256 
#define NLOOPS 1

#define M_PI 3.14159265358979323846
#define M_2PI 6.28318530717958647692


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

__device__  int find_index_f( float* array, int size, float value) {

int  low=0 ; 
int  high=size-1 ;
int  m_index= (high-low)/2 ;
while (m_index != low ) {
     if( value > array[m_index] ) low=m_index ; 
     else high=m_index ;  
       m_index=(high-low)/2 +low ;
}
return m_index ;

} 


__device__ void  rnd_to_fct2d(float& valuex,float& valuey,float rnd0,float rnd1, FH2D* hf2d) {


 int nbinsx=(*hf2d).nbinsx;
 int nbinsy=(*hf2d).nbinsy;
 float * HistoContents= (*hf2d).h_contents ;
 float* HistoBorders= (*hf2d).h_bordersx ;
 float* HistoBordersy= (*hf2d).h_bordersy ; 

 /*
 int ibin = nbinsx*nbinsy-1 ;
 for ( int i=0 ; i < nbinsx*nbinsy ; ++i) {
    if   (HistoContents[i]> rnd0 ) {
	 ibin = i ;
	 break ;
	}
 } 
*/
 int ibin=find_index_f(HistoContents, nbinsx*nbinsy, rnd0 ) ;


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


__device__ void ValidationHitSpy_d( Hit& hit,  const  Chain0_Args& args, unsigned long t, long long& pre_cell, bool is_second_spy ) {

  int cs = args.cs;
  //const int pdgId = args.pdgId;
  //double  charge   = args.charge;  

  long long cell = getDDE(args.geo, cs, hit.eta(), hit.phi()); // cs < 20
  
//  if(!is_second_spy) printf("HitSpy cells: tid=%d, pre_cell=%lu, cell=%lu, eta=%f, phi=%f\n", threadIdx.x+blockIdx.x*blockDim.x, pre_cell, cell, hit.eta(), hit.phi()  ) ;  

  bool is_matched =false ;
  if ( (cell == pre_cell)  && is_second_spy  )  is_matched = true ;
  CaloDetDescrElement *cellele = &(args.geo->cells[cell]) ;
  //long long id = cell.identify();
  //float eta=cell.eta();
  //float phi=cell.phi();
  //float z=cell.z();
  //float r=cell.r() ;

    pre_cell= cell ;


  double dphi_hit = hit.phi() - cellele->phi();
  while (dphi_hit > M_PI) {
    dphi_hit -= M_2PI;
  }
  while (dphi_hit <= -M_PI) {
    dphi_hit += M_2PI;
  }
 

  //float hitenergy = hit.E();

  Hitspy_Hist hspy = is_second_spy ? args.hs2 : args.hs1 ;
     
  short  ibin = hspy.hist_hitgeo_dphi.find_bin(dphi_hit) ;
  float *  x_ptr= hspy.hist_hitgeo_dphi.x_ptr ;
  short *   i_ptr= hspy.hist_hitgeo_dphi.i_ptr ;


  if(t<args.nhits) {
	 x_ptr[t]=dphi_hit ;
  	i_ptr[t]=ibin ;
  }

// This assummed  hist_hitgeo_matchprevious_dphi has same low, up and nbins 

  if( is_second_spy ) {
     bool *   match = hspy.hist_hitgeo_matchprevious_dphi.match ;

     if (t < args.nhits ) match[t]=is_matched ;
  }

//  bins.m_hist_hitgeo_dphi.set(dphi_hit, hit.E());

/*
  float extrapol_phi = hit.center_phi();
  float extrapol_r   = hit.center_r();
  float extrapol_z   = hit.center_z();
  float extrapol_eta = hit.center_eta();
 



  if(cs < 21) {

    float deta_hit_minus_extrapol = hit.eta() - extrapol_eta;
    float dphi_hit_minus_extrapol = hit.phi() - extrapol_phi;
    while (dphi_hit_minus_extrapol >= M_PI) dphi_hit_minus_extrapol -= M_2PI;
    while (dphi_hit_minus_extrapol < -M_PI) dphi_hit_minus_extrapol += M_2PI;

    if(charge < 0.0) dphi_hit_minus_extrapol = -dphi_hit_minus_extrapol;
    if(extrapol_eta < 0.0) deta_hit_minus_extrapol = -deta_hit_minus_extrapol;

    double m_deta_hit_minus_extrapol_mm = deta_hit_minus_extrapol * fabs( 2.0*exp(-extrapol_eta) / (1.0 + exp(-2.0*extrapol_eta)) ) * sqrt(extrapol_r*extrapol_r + extrapol_z*extrapol_z);
    double m_dphi_hit_minus_extrapol_mm = dphi_hit_minus_extrapol * extrapol_r;
    
    float alpha_mm = atan2(m_dphi_hit_minus_extrapol_mm, m_deta_hit_minus_extrapol_mm);


    bins.m_hist_deltaEta.set(m_deta_hit_minus_extrapol_mm, hitenergy);
    bins.m_hist_deltaPhi.set(m_dphi_hit_minus_extrapol_mm, hitenergy);
    bins.m_hist_deltaRt.set(hit.r() - extrapol_r, hitenergy);
    bins.m_hist_deltaZ.set(hit.z() - extrapol_z, hitenergy);


    ///////////////////////////////

    float alpha_absPhi_mm = atan2(fabs(m_dphi_hit_minus_extrapol_mm), m_deta_hit_minus_extrapol_mm);
    float radius_mm = sqrt(m_dphi_hit_minus_extrapol_mm * m_dphi_hit_minus_extrapol_mm + m_deta_hit_minus_extrapol_mm * m_deta_hit_minus_extrapol_mm);

    if (alpha_mm < 0) alpha_mm = M_2PI + alpha_mm;

    if(layer_energy > 0) {
      if (hitenergy < 0) hitenergy = 0;
      bins.m_hist_hitenergy_alpha_radius.set(alpha_mm, radius_mm, hitenergy / layer_energy);
      bins.m_hist_hitenergy_alpha_absPhi_radius.set(alpha_absPhi_mm, radius_mm, hitenergy / layer_energy);
    }
  }






  //////////////////////////////////////////////////////////////////////////////



  // Start of wiggle efficiency
  if (is_consider_eta_boundary) { // for layers where phi granularity changes at some eta
    float cell_eta = cellele->eta();
    float cell_deta = cellele->deta();

    // do not consider the cells that lie across this eta boundary
    if ( fabs(cell_eta) < eta_boundary && (fabs(cell_eta) + 0.5 * cell_deta) < eta_boundary) {
      bins.m_hist_total_hitPhi_minus_cellPhi.set(dphi_hit, hitenergy);
      if(is_matched) bins.m_hist_matched_hitPhi_minus_cellPhi.set(dphi_hit, hitenergy);
    } else if ( fabs(cell_eta) > eta_boundary && (fabs(cell_eta) - 0.5 * cell_deta) > eta_boundary) {
      bins.m_hist_total_hitPhi_minus_cellPhi_etaboundary.set(dphi_hit, hitenergy);
      if (is_matched) bins.m_hist_matched_hitPhi_minus_cellPhi_etaboundary.set(dphi_hit, hitenergy);
    }
  } else { // for layers there is no change in phi granularity
    bins.m_hist_total_hitPhi_minus_cellPhi.set(dphi_hit, hitenergy);
    if (is_matched) bins.m_hist_matched_hitPhi_minus_cellPhi.set(dphi_hit, hitenergy);
  }
  // End of wiggle efficiency




  bins.m_hist_Rz.set(hit.r(),hit.z(),hitenergy);

  // Start of m_hist_hitenergy_weight
  float w = 0.0;
  if ( args.isBarrel ){ // Barrel: weight from r
    w = (hit.r() - args.extrapol_r_ent)/(args.extrapol_r_ext - args.extrapol_r_ent);
  } else { // End-Cap and FCal: weight from z
    w = (hit.z() - args.extrapol_z_ent)/(args.extrapol_z_ext - args.extrapol_z_ent);
  }
  //if(m_hist_Rz_outOfRange && (w<0. || w>1.0) ) m_hist_Rz_outOfRange->Fill(hit.r(),hit.z());
  bins.m_hist_hitenergy_weight.set(w,hitenergy);
  //if( (cs!=3) && (w<=-0.25 || w >=1.25) ) ATH_MSG_DEBUG("Found weight outside [-0.25,1.25]: weight=" << w); // Weights are expected out of range in EMB3 (cs==3)
  // End of m_hist_hitenergy_weight





  if (hit.r() > args.extrapol_r_ent) bins.m_hist_hitenergy_r.set(hit.r(), hitenergy);

  if(fabs(hit.z()) > fabs(args.extrapol_z_ent)) bins.m_hist_hitenergy_z.set(hit.z(), hitenergy);


  bins.m_hist_hitgeo_matchprevious_dphi.set(dphi_hit, hit.E());
*/


}

__device__ void HistoLateralShapeParametrization_d( Hit& hit, unsigned long t, Chain0_Args args ) {

  //int     pdgId    = args.pdgId;
  float  charge   = args.charge;

  //int cs=args.charge;
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

#include "kern_main_chain.cu"



__global__  void simulate_chain0_A( float E, int nhits,  Chain0_Args args ) {

  int tid = threadIdx.x + blockIdx.x*blockDim.x ;
  for ( int i=0 ; i<NLOOPS ; ++i ) { 
    unsigned long t = tid+i*gridDim.x*blockDim.x ;
    if ( t  >= nhits ) break ; 
    Hit hit ;
    hit.E()=E ;
    CenterPositionCalculation_d( hit, args) ;
    long long pre_cell = -5 ;
    //if(args.spy) ValidationHitSpy_d( hit,  args, t, pre_cell, false ) ;
    HistoLateralShapeParametrization_d(hit,t,  args) ;
    if(args.spy) ValidationHitSpy_d( hit,  args, t, pre_cell, false ) ;
    HitCellMappingWiggle_d ( hit, args, t ) ;
    if(args.spy) ValidationHitSpy_d( hit,  args, t , pre_cell,  true ) ;
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




__host__  void *  CaloGpuGeneral::Rand4Hits_init( long long maxhits, unsigned short maxbin, unsigned long long seed ){ 

      Rand4Hits * rd4h = new Rand4Hits ;
	float * f  ;
	curandGenerator_t gen ;
       gpuQ(cudaMalloc((void**)&f , 3*maxhits*sizeof(float))) ;
        
        CURAND_CALL(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));
        CURAND_CALL(curandSetPseudoRandomGeneratorSeed(gen, seed)) ;

         rd4h->set_rand_ptr(f) ;
	 rd4h->set_gen(gen) ;

	std::cout<< "Allocating Hist in Rand4Hit_init()"<<std::endl;
	rd4h->allocate_hist(maxhits,maxbin,2000, 3, 1 ) ; 

	return  (void* ) rd4h ;

}

__host__  void   CaloGpuGeneral::Rand4Hits_finish( void * rd4h ){ 

 if ( (Rand4Hits *)rd4h ) delete (Rand4Hits *)rd4h  ;

}


__global__  void simulate_chain0_clean(Chain0_Args args) {
 unsigned long  tid = threadIdx.x + blockIdx.x*blockDim.x ;
 if(tid < args.ncells ) args.hitcells_b[tid]= false ;
 if(tid < args.maxhitct) {
    args.hitcells_l[tid] = 0 ;
    args.hitcounts_b[tid] =0 ; 
 }
 if(tid ==0 ) args.hitcells_ct[0]= 0 ; 
}


__global__ void simulate_chain0_block_hist(unsigned int * hitct_b, unsigned int ct , Chain0_Args args) {
extern __shared__ unsigned long hitcells[] ;
unsigned int *  counts = (unsigned int * ) (& hitcells[ct]) ;
 unsigned long  tid = threadIdx.x + blockIdx.x*blockDim.x ;
// unsigned long  hitcell_index=args.hitcells[tid];   //cell index for tid's hit

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
 unsigned long  hitcell_index=args.hitcells[tid];   //cell index for tid's hit
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
//if(threadIdx.x==0 ) for( int ii=0 ; ii<2 ; ii++) {printf("from Block-kernel block %d  counts[%d]=%d\n",blockIdx.x ,ii, counts[ii] ) ;}

__syncthreads() ;
for(int j =0 ; j< (ct+blockDim.x -1)/blockDim.x ; ++j ) {
	int index=threadIdx.x+j*blockDim.x;
	if(index < ct ) 
	hitct_b[ct* blockIdx.x+index]=counts[index] ;
	__syncthreads() ;
}


}

#include "kern_hits_counts.cu"

__device__ void warpReduce(volatile int* sdata, int tid) {
sdata[tid] += sdata[tid + 32];
sdata[tid] += sdata[tid + 16];
sdata[tid] += sdata[tid + 8];
sdata[tid] += sdata[tid + 4];
sdata[tid] += sdata[tid + 2];
sdata[tid] += sdata[tid + 1];
}


__global__ void simulate_chain0_hist_merge(unsigned int * hitct_b, int ct_blks, unsigned int ct , Chain0_Args & args) {
extern __shared__ int sdata[] ;
int tid=threadIdx.x;

if((tid+blockDim.x) < ct_blks ){
sdata[tid]=hitct_b[tid*ct+blockIdx.x]+hitct_b[(tid+blockDim.x)*ct+blockIdx.x] ;
}else{
sdata[tid] = (tid < ct_blks) ? hitct_b[tid*ct+blockIdx.x] : 0 ;  //protect when ct_blk<32 
}
__syncthreads();


for (unsigned int s=blockDim.x/2; s>32; s>>=1) {
if (tid < s)
sdata[tid] += sdata[tid + s];
__syncthreads();
}
if (tid < 32) warpReduce(sdata, tid);
if(tid==0)  hitct_b[blockIdx.x] =sdata[0] ;
//if(tid==0)  printf("block %d count=%d\n", blockIdx.x, sdata[0] ) ;

}



__host__ int highestPowerof2(unsigned int n) 
{ 
    // Invalid input 
    if (n < 1) 
        return 0; 
  
    int res = 1; 
  
    // Try all powers starting from 2^1 
    for (unsigned int i=0; i<8*sizeof(unsigned int); i++) 
    { 
        unsigned int curr = 1 << i; 
  
        // If current power is more than n, break 
        if (curr > n) 
           break; 
  
        res = curr; 
    } 
  
    return res; 
}


#include "kern_hitspy_hist.cu"

__host__ void CaloGpuGeneral::simulate_hits(float E, int nhits, Chain0_Args & args ) {

        Rand4Hits * rd4h = (Rand4Hits *) args.rd4h ;
	
	
        float * r= rd4h->rand_ptr()  ;
         args.rand = r ;
  CURAND_CALL(curandGenerateUniform(rd4h->gen(), r, 3*nhits));


	 
	unsigned long  ncells = args.ncells ; 
	args.maxhitct=MAXHITCT;



/*
        if(0) {
	gpuQ(cudaMalloc((void**)&(args.hitcells_b), args.ncells*sizeof(bool)));
	gpuQ(cudaMalloc((void**)&(args.hitcells_l), args.nhits*sizeof(unsigned long)));
	gpuQ(cudaMalloc((void**)&(args.hitcells), args.nhits*sizeof(unsigned long)));
	gpuQ(cudaMalloc((void**)&(args.hitcells_ct), sizeof(unsigned int)));
	if (args.spy) {
	gpuQ(cudaMalloc((void**)&(args.hs1.hist_hitgeo_dphi.x_ptr), sizeof(float)*args.nhits )) ;
	gpuQ(cudaMalloc((void**)&(args.hs1.hist_hitgeo_dphi.i_ptr), sizeof(short)*args.nhits )) ;
	gpuQ(cudaMalloc((void**)&(args.hs2.hist_hitgeo_dphi.x_ptr), sizeof(float)*args.nhits )) ;
	gpuQ(cudaMalloc((void**)&(args.hs2.hist_hitgeo_dphi.i_ptr), sizeof(short)*args.nhits )) ;
	gpuQ(cudaMalloc((void**)&(args.hs2.hist_hitgeo_matchprevious_dphi.match), sizeof(bool)*args.nhits )) ;

	// assume only 2 stages, nblocks of first stage < 1024 ,  each histogram  need store 1st stage output nblock*nbin 
	gpuQ(cudaMalloc((void**)&(args.hs1.hist_hitgeo_dphi.hb_ptr), sizeof(int)*args.hs1.hist_hitgeo_dphi.nbin*1024 )) ;
	gpuQ(cudaMalloc((void**)&(args.hs2.hist_hitgeo_dphi.hb_ptr), sizeof(int)*args.hs2.hist_hitgeo_dphi.nbin*1024 )) ;
	gpuQ(cudaMalloc((void**)&(args.hs2.hist_hitgeo_matchprevious_dphi.hb_ptr), sizeof(int)*args.hs2.hist_hitgeo_matchprevious_dphi.nbin*1024 )) ;

        // for store sumx sumx2  for each histogram
	gpuQ(cudaMalloc((void**)&(args.hs_sumx), sizeof(float)*1024*6 )) ;
	
	args.hs1.hist_hitgeo_dphi.ct_array=(int* ) malloc(args.hs1.hist_hitgeo_dphi.nbin*sizeof(int) ) ;
	args.hs2.hist_hitgeo_dphi.ct_array=(int* ) malloc(args.hs2.hist_hitgeo_dphi.nbin*sizeof(int) ) ;
	args.hs2.hist_hitgeo_matchprevious_dphi.ct_array=(int* ) malloc(args.hs2.hist_hitgeo_matchprevious_dphi.nbin*sizeof(int) ) ;

        
  
	} //if(spy)

	} //if(0/1) 
*/


args.hitcells_b=rd4h->get_B_ptrs()[0] ;
args.hitcells=rd4h->get_Ul_ptrs()[0] ;    //maxhit
args.hitcells_l=rd4h->get_Ul_ptrs()[1] ; //maxhitct
args.hitcells_ct= rd4h->get_Ui_ptrs()[0] ; //single value, number of  uniq hit cells

//std::cout<<"hitcells_b: " << args.hitcells_b << ", args.hitcells=" << args.hitcells 
//    << ", args.hitcells_l="<< args.hitcells_l << ", args.hitcells_ct=" << args.hitcells_ct << std::endl ;


args.hitcounts_b= rd4h->get_Ui_ptrs()[1] ;  // block wise counts results

args.hitcells_h = rd4h->get_hitcells() ; //host hit cells indexes
args.hitcells_ct_h =  rd4h->get_hitcells_ct() ;

if(args.spy) {
args.hs1.hist_hitgeo_dphi.x_ptr=rd4h->get_F_ptrs()[0] ;
args.hs2.hist_hitgeo_dphi.x_ptr=rd4h->get_F_ptrs()[1] ;
args.hs2.hist_hitgeo_matchprevious_dphi.x_ptr=args.hs2.hist_hitgeo_dphi.x_ptr ;
args.hs1.hist_hitgeo_dphi.i_ptr=rd4h->get_S_ptrs()[0] ;      
args.hs2.hist_hitgeo_dphi.i_ptr=rd4h->get_S_ptrs()[1] ;      
args.hs2.hist_hitgeo_matchprevious_dphi.i_ptr=args.hs2.hist_hitgeo_dphi.i_ptr;      

args.hs1.hist_hitgeo_dphi.hb_ptr=rd4h->get_I_ptrs()[0] ;
args.hs2.hist_hitgeo_dphi.hb_ptr=rd4h->get_I_ptrs()[1] ;
args.hs2.hist_hitgeo_matchprevious_dphi.hb_ptr=rd4h->get_I_ptrs()[2] ;

args.hs_sumx=rd4h->get_F_ptrs()[2] ;

args.hs2.hist_hitgeo_matchprevious_dphi.match=rd4h->get_B_ptrs()[1] ;


args.hs1.hist_hitgeo_dphi.ct_array_g=rd4h->get_D_ptrs()[0];
args.hs2.hist_hitgeo_dphi.ct_array_g=rd4h->get_D_ptrs()[1];
args.hs2.hist_hitgeo_matchprevious_dphi.ct_array_g=rd4h->get_D_ptrs()[2];

args.hs1.hist_hitgeo_dphi.sumw2_array_g=rd4h->get_D_ptrs()[3];
args.hs2.hist_hitgeo_dphi.sumw2_array_g=rd4h->get_D_ptrs()[4];
args.hs2.hist_hitgeo_matchprevious_dphi.sumw2_array_g=rd4h->get_D_ptrs()[5];
args.hs_sumwx_g=rd4h->get_D_ptrs()[6] ;

args.hs_nentries=rd4h->get_Ull_ptrs()[0] ;
args.hs_sumwx_h=rd4h->get_hist_stat_h() ;

args.hs1.hist_hitgeo_dphi.ct_array_h=rd4h-> get_array_h_ptrs()[0] ;
args.hs2.hist_hitgeo_dphi.ct_array_h=rd4h-> get_array_h_ptrs()[1] ;
args.hs2.hist_hitgeo_matchprevious_dphi.ct_array_h=rd4h-> get_array_h_ptrs()[2] ;
args.hs1.hist_hitgeo_dphi.sumw2_array_h=rd4h-> get_sumw2_array_h_ptrs()[0] ;
args.hs2.hist_hitgeo_dphi.sumw2_array_h=rd4h-> get_sumw2_array_h_ptrs()[1] ;
args.hs2.hist_hitgeo_matchprevious_dphi.sumw2_array_h=rd4h-> get_sumw2_array_h_ptrs()[2] ;



}


 	cudaError_t err = cudaGetLastError();

   if(args.is_first && args.spy) {


//	std::cout<<" first event , memset 0 the hitspy histgrams storages on GPU " << std::endl ;
//	std::cout<< "Pointers: " << std::endl 
//		<<args.hs1.hist_hitgeo_dphi.ct_array_g <<std::endl  
//		<<args.hs1.hist_hitgeo_dphi.sumw2_array_g <<std::endl  
//		<<args.hs2.hist_hitgeo_dphi.ct_array_g <<std::endl  
//		<<args.hs2.hist_hitgeo_dphi.sumw2_array_g <<std::endl  
		; 
	gpuQ(cudaMemset(args.hs1.hist_hitgeo_dphi.ct_array_g, 0 ,(args.hs1.hist_hitgeo_dphi.nbin+2)*sizeof(double) )) ;
	gpuQ(cudaMemset(args.hs1.hist_hitgeo_dphi.sumw2_array_g, 0 ,(args.hs1.hist_hitgeo_dphi.nbin+2)*sizeof(double)) ) ;

	gpuQ(cudaMemset(args.hs2.hist_hitgeo_dphi.ct_array_g, 0 ,(args.hs2.hist_hitgeo_dphi.nbin+2)*sizeof(double)) ) ;
	gpuQ(cudaMemset(args.hs2.hist_hitgeo_dphi.sumw2_array_g, 0 ,(args.hs2.hist_hitgeo_dphi.nbin+2)*sizeof(double)) ) ;

	gpuQ(cudaMemset(args.hs2.hist_hitgeo_matchprevious_dphi.ct_array_g, 0 ,(args.hs2.hist_hitgeo_matchprevious_dphi.nbin+2)*sizeof(double)) ) ;
	gpuQ(cudaMemset(args.hs2.hist_hitgeo_matchprevious_dphi.sumw2_array_g, 0 ,(args.hs2.hist_hitgeo_matchprevious_dphi.nbin+2)*sizeof(double)) ) ;

	gpuQ(cudaMemset(args.hs_sumwx_g,0, 12*sizeof(double))) ;
	gpuQ(cudaMemset(args.hs_nentries,0, 3*sizeof(unsigned long long ))) ;


  } 
        
	int blocksize=BLOCK_SIZE ;
	int threads_tot= args.ncells  ;
	int nblocks= (threads_tot + blocksize-1 )/blocksize ;        


	simulate_chain0_clean <<< nblocks, blocksize >>>( args) ;
// 	cudaDeviceSynchronize() ;
// if (err != cudaSuccess) {
//        std::cout<< "simulate_chain0_clean "<<cudaGetErrorString(err)<< std::endl;
//}


//args.spy=false ;
	blocksize=BLOCK_SIZE ;
	threads_tot= (nhits +NLOOPS-1) /NLOOPS  ;
	nblocks= (threads_tot + blocksize-1 )/blocksize ;        

//	 std::cout<<"Nblocks: "<< nblocks << ", blocksize: "<< blocksize 
 //               << ", total Threads: " << threads_tot << std::endl ;


  int fh_size=args.fh2d_v.nbinsx+args.fh2d_v.nbinsy+2+(args.fh2d_v.nbinsx+1)*(args.fh2d_v.nbinsy+1) ;
 if(args.debug) std::cout<<"2DHisto_Func_size: " << args.fh2d_v.nbinsx << ", " << args.fh2d_v.nbinsy << "= " << fh_size <<std::endl ; 
if(fh_size > 600) 
 simulate_chain0_A <<<nblocks, blocksize  >>> (E, nhits, args  ) ;
else  simulate_chain0_A_sh <<<nblocks, blocksize, fh_size*sizeof(float)   >>> (E, nhits, args ) ; 

  cudaDeviceSynchronize() ;
  err = cudaGetLastError();
 if (err != cudaSuccess) {
        std::cout<< "simulate_chain0_A "<<cudaGetErrorString(err)<< std::endl;
}


  nblocks = (ncells + blocksize -1 )/blocksize ;
  simulate_chain0_B1 <<<nblocks,blocksize >>> (args) ;
  cudaDeviceSynchronize() ;
 err = cudaGetLastError();
 if (err != cudaSuccess) {
        std::cout<< "simulate_chain0_B1 "<<cudaGetErrorString(err)<< std::endl;
}

   unsigned int ct ;
   gpuQ(cudaMemcpy(&ct, args.hitcells_ct,sizeof(unsigned int), cudaMemcpyDeviceToHost));
   unsigned long *hitcells =args.hitcells_h;
   int * hitcells_ct =args.hitcells_ct_h ;
//   hitcells=(unsigned long * ) malloc(sizeof(unsigned long)*ct) ; //list of hit cells
 //  int * hitcells_ct=(int * ) malloc(sizeof(int )*ct) ;    
 // check result 
   gpuQ(cudaMemcpy(hitcells, args.hitcells_l, ct*sizeof(unsigned long), cudaMemcpyDeviceToHost));

 if(args.debug)	std::cout<<"hit cell counts="<<ct<<std::endl;
//for (int tt=0; tt<ct ; tt++)
//	std::cout<<"hit cell ["<<tt<<"]="<<hitcells[tt]<<std::endl;


   blocksize= highestPowerof2(args.nhits)/512 ;
   if(blocksize <32 ) blocksize=32;
   nblocks = (args.nhits + blocksize-1)/blocksize ;
//std::cout<<"blocksize="<<blocksize << " Nhits="<<args.nhits << " ct="<<ct<<" nblocks="<<nblocks<< std::endl ;  
//	unsigned int * hitcounts_b ;
//	gpuQ(cudaMalloc((void**)&(hitcounts_b), ct*nblocks*sizeof(unsigned int)));

// std::cout<<"nblocks for hit counts="<<nblocks<< ", blocksize="<<blocksize<<std::endl ;

if(args.nhits < 100000) {
    simulate_chain0_hit_ct_small <<<nblocks, blocksize, ct*(sizeof(unsigned long)+sizeof(unsigned int)) >>>(args.hitcounts_b,ct,args) ;
 cudaDeviceSynchronize() ;
 err = cudaGetLastError();
 if (err != cudaSuccess) {
        std::cout<< "simulate_chain0_hit_ct_small "<<cudaGetErrorString(err)<< std::endl;
 }

} else {

   simulate_chain0_block_hist<<<nblocks, blocksize,ct*(sizeof(unsigned long)+sizeof(unsigned int))>>> (args.hitcounts_b,ct,args) ;
  cudaDeviceSynchronize() ;
 err = cudaGetLastError();
 if (err != cudaSuccess) {
        std::cout<< "simulate_chain0_block_hist "<<cudaGetErrorString(err)<< std::endl;
 }
   

    int ct_b = nblocks ;  
    blocksize=highestPowerof2(ct_b-1); // when ct_b is 2^n need half of it as block size
    if(blocksize <64)  blocksize=64 ; // warpreduce only works with block size >=64, 2^n
    nblocks=ct;

//	std::cout<<"merge Block size,nblocks="<< blocksize<< " " <<nblocks <<std::endl ;

    simulate_chain0_hist_merge<<<nblocks,blocksize,blocksize*sizeof(unsigned int) >>>(args.hitcounts_b,ct_b,ct,args) ;
  cudaDeviceSynchronize() ;
 err = cudaGetLastError();
 if (err != cudaSuccess) {
        std::cout<< "simulate_chain0_hist_merge "<<cudaGetErrorString(err)<< std::endl;
 }

}
   gpuQ(cudaMemcpy(hitcells_ct, args.hitcounts_b, ct*sizeof(int), cudaMemcpyDeviceToHost));

// pass result back 
   args.ct=ct;
//   args.hitcells_h=hitcells ;
//   args.hitcells_ct_h=hitcells_ct ;

/*
int total_ct=0 ;
for(int ii =0 ; ii<ct ; ++ii) {
	std::cout<< "CT["<<hitcells[ii]<<"]=" << hitcells_ct[ii]<<std::endl ;
	total_ct += hitcells_ct[ii] ;
}
std::cout << "Total Counts =" << total_ct << " nhits=" << args.nhits<< std::endl;

*/
if(args.spy) {
   blocksize= highestPowerof2(args.nhits)/512 ;
   if(blocksize <32 ) blocksize=32;
   nblocks = (args.nhits + blocksize-1)/blocksize ;
   int nbins= args.hs1.hist_hitgeo_dphi.nbin+2 ;

//std::cout<< "nbin= "<<nbins << ", blocksize= " << blocksize 
//	<< ",nblock= "<<nblocks << ", args.hs1.hist_hitgeo_dphi.i_ptr=" << args.hs1.hist_hitgeo_dphi.i_ptr
//	<< ", args.nhits" << args.nhits << ", args.hs1.hist_hitgeo_dphi.hb_ptr="<<args.hs1.hist_hitgeo_dphi.hb_ptr
//	<< std::endl ; 

// hs1
   hitspy_hist_stgA<<< nblocks, blocksize, sizeof(int)* nbins >>> 
       (args.hs1.hist_hitgeo_dphi.i_ptr, nbins, args.nhits,
		args.hs1.hist_hitgeo_dphi.hb_ptr)       ;


   int nbsz=highestPowerof2(nblocks-1) ;
   if (nbsz < 64) nbsz=64 ; 
   hitspy_hist_stgB<<< nbins, nbsz, sizeof(int)*nbsz >>> 
	(args.hs1.hist_hitgeo_dphi.hb_ptr, args.hs1.hist_hitgeo_dphi.ct_array_g ,args.hs1.hist_hitgeo_dphi.sumw2_array_g, &args.hs_nentries[0],
		 &args.hs_sumwx_g[0], nblocks,  nbins,E, false ,args.nhits ) ;
//   gpuQ(cudaMemcpy(args.hs1.hist_hitgeo_dphi.ct_array, args.hs1.hist_hitgeo_dphi.hb_ptr, nbins*sizeof(int), cudaMemcpyDeviceToHost));

  cudaDeviceSynchronize() ;
 err = cudaGetLastError();
 if (err != cudaSuccess) {
        std::cout<< "hitspy_hist_stgB "<<cudaGetErrorString(err)<< std::endl;
}

//std::cout<<"E="<<E<< std::endl ;

//hs2
    nbins = args.hs2.hist_hitgeo_dphi.nbin+2 ;
   hitspy_hist_stgA<<< nblocks, blocksize, sizeof(int)* nbins >>>
       (args.hs2.hist_hitgeo_dphi.i_ptr, nbins, args.nhits,
                args.hs2.hist_hitgeo_dphi.hb_ptr)       ;
   hitspy_hist_stgB<<< nbins, nbsz, sizeof(int)*nbsz >>> 
	(args.hs2.hist_hitgeo_dphi.hb_ptr, args.hs2.hist_hitgeo_dphi.ct_array_g,args.hs2.hist_hitgeo_dphi.sumw2_array_g, &args.hs_nentries[1],
		 &args.hs_sumwx_g[4], nblocks,  nbins,E, false, args.nhits ) ;
// gpuQ(cudaMemcpy(args.hs2.hist_hitgeo_dphi.ct_array, args.hs2.hist_hitgeo_dphi.hb_ptr, nbins*sizeof(int), cudaMemcpyDeviceToHost));

  cudaDeviceSynchronize() ;
 err = cudaGetLastError();
 if (err != cudaSuccess) {
        std::cout<< "hitspy_hist_stgB "<<cudaGetErrorString(err)<< std::endl;
}

//hs2 matched 
    nbins = args.hs2.hist_hitgeo_matchprevious_dphi.nbin+2 ;
   hitspy_hist_matched_stgA<<< nblocks, blocksize, sizeof(int)* nbins >>>
       (args.hs2.hist_hitgeo_matchprevious_dphi.i_ptr,
	args.hs2.hist_hitgeo_matchprevious_dphi.match, 
	nbins, 
	args.nhits,
        args.hs2.hist_hitgeo_matchprevious_dphi.hb_ptr)       ;

  cudaDeviceSynchronize() ;
 err = cudaGetLastError();
 if (err != cudaSuccess) {
        std::cout<< "hitspy_hist_stgB "<<cudaGetErrorString(err)<< std::endl;
}
//int t[256*781];
// gpuQ(cudaMemcpy(t, args.hs2.hist_hitgeo_matchprevious_dphi.hb_ptr, 781*nbins*sizeof(int), cudaMemcpyDeviceToHost));

//for(int ii=0; ii<256*781 ; ii++) if(t[ii] !=0 )  std::cout<<"hb_ptr"<<ii<<"="<<t[ii] <<std::endl;


   hitspy_hist_stgB<<< nbins, nbsz, sizeof(int)*nbsz >>> 
	(args.hs2.hist_hitgeo_matchprevious_dphi.hb_ptr, args.hs2.hist_hitgeo_matchprevious_dphi.ct_array_g,
		args.hs2.hist_hitgeo_matchprevious_dphi.sumw2_array_g, &args.hs_nentries[2], 
		&args.hs_sumwx_g[8], nblocks,  nbins,E,true,args.nhits ) ;
  // gpuQ(cudaMemcpy(args.hs2.hist_hitgeo_matchprevious_dphi.ct_array, args.hs2.hist_hitgeo_matchprevious_dphi.hb_ptr, nbins*sizeof(int), cudaMemcpyDeviceToHost));
  cudaDeviceSynchronize() ;
 err = cudaGetLastError();
 if (err != cudaSuccess) {
        std::cout<< "hitspy_hist_stgB "<<cudaGetErrorString(err)<< std::endl;
}

//std::cout<<"3"<< std::endl ;

//sumx, sumx2
   nblocks = (args.nhits -1 )/(2*blocksize) +1 ;
   //if (nblocks <32)  nblocks =32 ;
   hitspy_hist_sumx_stgA<<< nblocks, blocksize,sizeof(float)* blocksize*2 >>>
	(  args.hs1.hist_hitgeo_dphi.x_ptr, nhits, args.hs_sumx,  &args.hs_sumx[1024]  ) ;
   nbsz = highestPowerof2(nblocks-1)   ; 
   if(nbsz < 64 ) nbsz=64 ;

//std::cout << "blocksize="<<nbsz <<" , nblocks="<<nblocks <<  std::endl ;

  hitspy_hist_sumx_stgB<<<1, nbsz, sizeof(float)*nbsz*2 >>> (  nblocks, &args.hs_sumx[0],   &args.hs_sumx[1024],&args.hs_sumwx_g[0],E  );
 //  gpuQ(cudaMemcpy(&args.hs1.hist_hitgeo_dphi.sumx, &args.hs_sumx[0], sizeof(float), cudaMemcpyDeviceToHost));
 //  gpuQ(cudaMemcpy(&args.hs1.hist_hitgeo_dphi.sumx2, &args.hs_sumx[1024], sizeof(float), cudaMemcpyDeviceToHost));
    
   hitspy_hist_sumx_stgA<<< nblocks, blocksize,sizeof(float)* blocksize*2 >>>
	(  args.hs2.hist_hitgeo_dphi.x_ptr, nhits, &args.hs_sumx[2048],  &args.hs_sumx[3072]  ) ;
  hitspy_hist_sumx_stgB<<<1, nbsz, sizeof(float)*nbsz*2 >>> (  nblocks, &args.hs_sumx[2048],  &args.hs_sumx[3072],&args.hs_sumwx_g[4],E  ) ;
 //  gpuQ(cudaMemcpy(&args.hs2.hist_hitgeo_dphi.sumx, &args.hs_sumx[2048], sizeof(float), cudaMemcpyDeviceToHost));
//   gpuQ(cudaMemcpy(&args.hs2.hist_hitgeo_dphi.sumx2, &args.hs_sumx[3072], sizeof(float), cudaMemcpyDeviceToHost));
    

   hitspy_hist_sumx_matched_stgA<<< nblocks, blocksize,sizeof(float)* blocksize*2 >>>
	(  args.hs2.hist_hitgeo_matchprevious_dphi.x_ptr, 
	nhits, 
	&args.hs_sumx[4096],  
	&args.hs_sumx[5120], 
	args.hs2.hist_hitgeo_matchprevious_dphi.match) ;
  hitspy_hist_sumx_stgB<<<1, nbsz, sizeof(float)*nbsz*2 >>> (  nblocks, &args.hs_sumx[4096],  &args.hs_sumx[5120],&args.hs_sumwx_g[8], E  ) ;
  // gpuQ(cudaMemcpy(&args.hs2.hist_hitgeo_matchprevious_dphi.sumx, &args.hs_sumx[4096], sizeof(float), cudaMemcpyDeviceToHost));
  // gpuQ(cudaMemcpy(&args.hs2.hist_hitgeo_matchprevious_dphi.sumx2, &args.hs_sumx[5120], sizeof(float), cudaMemcpyDeviceToHost));
    

}

  cudaDeviceSynchronize() ;
 err = cudaGetLastError();
 if (err != cudaSuccess) {
        std::cout<< "hitspy_hist_stgB "<<cudaGetErrorString(err)<< std::endl;
}


if(args.is_last && args.spy) {
  //std::cout<<" Last event" <<std::endl ;

  double *  buffer = args.hs_sumwx_h ;
  gpuQ(cudaMemcpy(buffer, args.hs_sumwx_g, 12*sizeof(double), cudaMemcpyDeviceToHost));
  args.hs1.hist_hitgeo_dphi.sumx_h=buffer[0] ;
  args.hs1.hist_hitgeo_dphi.sumx2_h=buffer[1] ;
  args.hs1.hist_hitgeo_dphi.sumw_h=buffer[2] ;
  args.hs1.hist_hitgeo_dphi.sumw2_h=buffer[3] ;

  args.hs2.hist_hitgeo_dphi.sumx_h=buffer[4] ;
  args.hs2.hist_hitgeo_dphi.sumx2_h=buffer[5] ;
  args.hs2.hist_hitgeo_dphi.sumw_h=buffer[6] ;
  args.hs2.hist_hitgeo_dphi.sumw2_h=buffer[7] ;

  args.hs2.hist_hitgeo_matchprevious_dphi.sumx_h=buffer[8] ;
  args.hs2.hist_hitgeo_matchprevious_dphi.sumx2_h=buffer[9] ;
  args.hs2.hist_hitgeo_matchprevious_dphi.sumw_h=buffer[10] ;
  args.hs2.hist_hitgeo_matchprevious_dphi.sumw2_h=buffer[11] ;

  unsigned long long nentry_buffer[3] ;
  gpuQ(cudaMemcpy(nentry_buffer, args.hs_nentries, 3*sizeof(unsigned long long), cudaMemcpyDeviceToHost));
  args.hs1.hist_hitgeo_dphi.nentries=nentry_buffer[0] ;
  args.hs2.hist_hitgeo_dphi.nentries=nentry_buffer[1] ;
  args.hs2.hist_hitgeo_matchprevious_dphi.nentries=nentry_buffer[2] ;

//int nbins = args.hs2.hist_hitgeo_matchprevious_dphi.nbin+2;
//std::cout<< "Nentries for histSpy histo="<< nentry_buffer[0]<<", "<< nentry_buffer[1] << ", " << nentry_buffer[2] <<" "<< nbins<< std::endl ;
//for (int jj=0 ;jj<12 ;jj++) 
//std::cout<<"scaler Buffer["<<jj<<"]="<<buffer[jj]<<std::endl;

  gpuQ(cudaMemcpy(args.hs1.hist_hitgeo_dphi.ct_array_h, args.hs1.hist_hitgeo_dphi.ct_array_g,
	(args.hs1.hist_hitgeo_dphi.nbin+2)*sizeof(double), cudaMemcpyDeviceToHost));
  gpuQ(cudaMemcpy(args.hs2.hist_hitgeo_dphi.ct_array_h, args.hs2.hist_hitgeo_dphi.ct_array_g, 
	(args.hs2.hist_hitgeo_dphi.nbin+2)*sizeof(double), cudaMemcpyDeviceToHost));
  gpuQ(cudaMemcpy(args.hs2.hist_hitgeo_matchprevious_dphi.ct_array_h,
	 args.hs2.hist_hitgeo_matchprevious_dphi.ct_array_g, 
	(args.hs2.hist_hitgeo_matchprevious_dphi.nbin+2)*sizeof(double), cudaMemcpyDeviceToHost));

  gpuQ(cudaMemcpy(args.hs1.hist_hitgeo_dphi.sumw2_array_h, args.hs1.hist_hitgeo_dphi.sumw2_array_g,
	(args.hs1.hist_hitgeo_dphi.nbin+2)*sizeof(double), cudaMemcpyDeviceToHost));
  gpuQ(cudaMemcpy(args.hs2.hist_hitgeo_dphi.sumw2_array_h, args.hs2.hist_hitgeo_dphi.sumw2_array_g, 
	(args.hs2.hist_hitgeo_dphi.nbin+2)*sizeof(double), cudaMemcpyDeviceToHost));
  gpuQ(cudaMemcpy(args.hs2.hist_hitgeo_matchprevious_dphi.sumw2_array_h,
	 args.hs2.hist_hitgeo_matchprevious_dphi.sumw2_array_g, 
	(args.hs2.hist_hitgeo_matchprevious_dphi.nbin+2)*sizeof(double), cudaMemcpyDeviceToHost));


//for(int ii=0; ii<nbins ; ++ii) {
//std::cout<<args.hs1.hist_hitgeo_dphi.ct_array_h[ii]<<", "<<args.hs2.hist_hitgeo_dphi.ct_array_h[ii]<<", "
//  	<<args.hs2.hist_hitgeo_matchprevious_dphi.ct_array_h[ii] <<"        "<< args.hs1.hist_hitgeo_dphi.sumw2_array_h[ii]<<", "
//	<<args.hs2.hist_hitgeo_dphi.sumw2_array_h[ii]<<", "<<args.hs2.hist_hitgeo_matchprevious_dphi.sumw2_array_h[ii]<< std::endl;

	
// need to take sqrt to enable TH1's  SetError() method ;
  for(int ii=0; ii<args.hs1.hist_hitgeo_dphi.nbin+2 ; ++ii) 
	args.hs1.hist_hitgeo_dphi.sumw2_array_h[ii]=sqrt(args.hs1.hist_hitgeo_dphi.sumw2_array_h[ii]);

  for(int ii=0; ii<args.hs2.hist_hitgeo_dphi.nbin+2 ; ++ii) 
	args.hs2.hist_hitgeo_dphi.sumw2_array_h[ii]=sqrt(args.hs2.hist_hitgeo_dphi.sumw2_array_h[ii]);

  for(int ii=0; ii<args.hs2.hist_hitgeo_matchprevious_dphi.nbin+2 ; ++ii) 
	args.hs2.hist_hitgeo_matchprevious_dphi.sumw2_array_h[ii]=sqrt(args.hs2.hist_hitgeo_matchprevious_dphi.sumw2_array_h[ii]);

  



}//if(args.spy)


}




