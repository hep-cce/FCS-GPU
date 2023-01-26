/*
  Copyright (C) 2002-2021 CERN for the benefit of the ATLAS collaboration
*/

#include "CaloGpuGeneral.h"
//#include "GeoRegion.cu"
#include "Hit.h"
#include "Rand4Hits.h"

//#include "gpuQ.h"
#include "Args.h"
#include "OMP_BigMem.h"

#include <chrono>
#include <mutex>
#include <omp.h>

static CaloGpuGeneral::KernelTime timing;

#define DEFAULT_BLOCK_SIZE 256
#define NLOOPS 1

static std::once_flag calledGetEnv {};
static int BLOCK_SIZE{DEFAULT_BLOCK_SIZE};

#define M_PI 3.14159265358979323846
#define M_2PI 6.28318530717958647692


//__device__  long long getDDE( GeoGpu* geo, int sampling, float eta, float phi) {
//
//   float * distance = 0 ;
//   int * steps =0 ;
//
//int MAX_SAMPLING = geo->max_sample ;
//Rg_Sample_Index * SampleIdx = geo->sample_index ;
// GeoRegion * regions_g = geo->regions ;
//
//if(sampling<0) return -1;
//  if(sampling>=MAX_SAMPLING) return -1;
//
//   int sample_size= SampleIdx[sampling].size ;
//   int sample_index=SampleIdx[sampling].index ;
//
//   GeoRegion * gr = ( GeoRegion *) regions_g ; 
//  if(sample_size==0) return -1;
//  float dist;
//  long long bestDDE=-1;
//  if(!distance) distance=&dist;
//  *distance=+10000000;
//  int intsteps;
//  int beststeps;
//  if ( steps )
//    beststeps = ( *steps );
//  else
//    beststeps = 0;
//  
//  if(sampling<21) {
//    for(int skip_range_check=0;skip_range_check<=1;++skip_range_check) {
//      for(unsigned int j= sample_index; j< sample_index+sample_size ; ++j) {
//        if(!skip_range_check) {
//          if(eta< gr[j].mineta()) continue;
//          if(eta> gr[j].maxeta()) continue;
//        }
//        if ( steps )
//          intsteps = ( *steps );
//         else 
//       intsteps=0;
//        float newdist;
//        long long  newDDE= gr[j].getDDE(eta,phi,&newdist,&intsteps);
//        if(newdist<*distance) {
//          bestDDE=newDDE;
//          *distance=newdist;
//          if(steps) beststeps=intsteps;
//          if(newdist<-0.1) break; //stop, we are well within the hit cell
//       }
//      }
//      if(bestDDE>=0) break;
//  }
//  } else {
//                return -3;
//  }
//  if(steps) *steps=beststeps;
//
//  return bestDDE;
//}
//
//
//__device__  int find_index_f( float* array, int size, float value) {
//// fist index (from 0)  have element value > value 
//// array[i] > value ; array[i-1] <= value 
//// std::upbund( )
//int  low=0 ; 
//int  high=size-1 ;
//int  m_index= (high-low)/2 ;
//while (m_index != high ) {
//    if ( value < array[m_index] )
//      high = m_index;
//    else
//      low = m_index + 1;
//       m_index=(high+low+1)/2 ;
//}
//return m_index ;
//
//} 
//
//
//
//__device__  int find_index_uint32( uint32_t* array, int size, uint32_t value) {
//// fist index i  have element value > value 
//// array[i] > value ; array[i-1] <= value
//int  low=0 ;
//int  high=size-1 ;
//int  m_index= (high-low)/2 ;
//while (m_index != high ) {
//    if ( value < array[m_index] )
//      high = m_index;
//    else
//      low = m_index + 1;
//       m_index=(high+low+1)/2  ;
//}
//return m_index ;
//
//}
//
//__device__  int find_index_long( long* array, int size, long value) {
//// find the first index of element which has vaule > value 
//int  low=0 ;
//int  high=size-1 ;
//int  m_index= (high-low)/2 ;
//while (high != low ) {
//    if ( value > array[m_index] )
//      low = m_index + 1;
//     else if( value == array[m_index] )  {
//        return m_index + 1   ;
//       // return min(m_index + 1, size-1)   ;
//    } else
//      high = m_index;
//       m_index=(high-low)/2 +low ;
//}
//return m_index ;
//
//}
//
//
//__device__ void  rnd_to_fct2d(float& valuex,float& valuey,float rnd0,float rnd1, FH2D* hf2d) {
//
//
// int nbinsx=(*hf2d).nbinsx;
// int nbinsy=(*hf2d).nbinsy;
// float * HistoContents= (*hf2d).h_contents ;
// float* HistoBorders= (*hf2d).h_bordersx ;
// float* HistoBordersy= (*hf2d).h_bordersy ; 
//
// /*
// int ibin = nbinsx*nbinsy-1 ;
// for ( int i=0 ; i < nbinsx*nbinsy ; ++i) {
//    if   (HistoContents[i]> rnd0 ) {
//	 ibin = i ;
//	 break ;
//	}
// } 
//*/
// int ibin=find_index_f(HistoContents, nbinsx*nbinsy, rnd0 ) ;
//
//
//  int biny = ibin/nbinsx;
//  int binx = ibin - nbinsx*biny;
//
//  float basecont=0;
//  if(ibin>0) basecont=HistoContents[ibin-1];
//
//  float dcont=HistoContents[ibin]-basecont;
//  if(dcont>0) {
//    valuex = HistoBorders[binx] + (HistoBorders[binx+1]-HistoBorders[binx]) * (rnd0-basecont) / dcont;
//  } else {
//    valuex = HistoBorders[binx] + (HistoBorders[binx+1]-HistoBorders[binx]) / 2;
//  }
//  valuey = HistoBordersy[biny] + (HistoBordersy[biny+1]-HistoBordersy[biny]) * rnd1;
//
//
//}
//
//
//__device__  float  rnd_to_fct1d( float  rnd, uint32_t* contents, float* borders , int nbins, uint32_t s_MaxValue  ) {
//
//
//  uint32_t int_rnd=s_MaxValue*rnd;
///*
//  int  ibin=nbins-1 ;
//  for ( int i=0 ; i < nbins ; ++i) {
//    if   (contents[i]> int_rnd ) {
//         ibin = i ;
//         break ;
//        }
//  }
//*/
//  int ibin=find_index_uint32(contents, nbins, int_rnd ) ;
//
//  int binx = ibin;
//
//  uint32_t basecont=0;
//  if(ibin>0) basecont=contents[ibin-1];
//
//  uint32_t dcont=contents[ibin]-basecont;
//  if(dcont>0) {
//    return borders[binx] + ((borders[binx+1]-borders[binx]) * (int_rnd-basecont)) / dcont;
//  } else {
//    return borders[binx] + (borders[binx+1]-borders[binx]) / 2;
//  }
//
//}
//




void *  CaloGpuGeneral::Rand4Hits_init( long long maxhits, int  maxbin, unsigned long long seed, bool hitspy ){ 

   auto t0 = std::chrono::system_clock::now();
      Rand4Hits * rd4h = new Rand4Hits ;
	float * f  ;
//	curandGenerator_t gen ;
   auto t1 = std::chrono::system_clock::now();
        
//        CURAND_CALL(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));
//        CURAND_CALL(curandSetPseudoRandomGeneratorSeed(gen, seed)) ;
//   auto t2 = std::chrono::system_clock::now();
//       gpuQ(cudaMalloc((void**)&f , 3*maxhits*sizeof(float))) ;
//   auto t3 = std::chrono::system_clock::now();
//         rd4h->set_rand_ptr(f) ;
//	 rd4h->set_gen(gen) ;
//	 rd4h->set_t_a_hits(maxhits);
//	 rd4h->set_c_hits(0) ;
//	CURAND_CALL(curandGenerateUniform(gen, f, 3*maxhits));
//   auto t4 = std::chrono::system_clock::now();
//
//	std::cout<< "Allocating Hist in Rand4Hit_init()"<<std::endl;
//	//rd4h->allocate_hist(maxhits,maxbin,2000, 3, 1, hitspy) ; 
//	rd4h->allocate_simulation(maxbin,MAXHITCT, MAX_CELLS) ; 
//	CU_BigMem * bm = new CU_BigMem(M_SEG_SIZE) ;
//        CU_BigMem::bm_ptr = bm ;
//   auto t5 = std::chrono::system_clock::now();
//
//  std::chrono::duration<double> diff1 = t1-t0 ;
//  std::chrono::duration<double> diff2 = t2-t1 ;
//  std::chrono::duration<double> diff3 = t3-t2 ;
//  std::chrono::duration<double> diff4 = t4-t3 ;
//  std::chrono::duration<double> diff5 = t5-t4 ;
///*  std::cout<<"Time of R4hit: " << diff1.count() << 
//     ","<< 
//       diff2.count() <<  
//     ","<< 
//       diff3.count() <<  
//     ","<< 
//       diff4.count() <<  
//     ","<< 
//       diff5.count() <<  " s" << std::endl ;
//*/
//
	return  (void* ) rd4h ;

}
//__host__  void   CaloGpuGeneral::Rand4Hits_finish( void * rd4h ){ 
//
//  size_t free, total ;
//  gpuQ(cudaMemGetInfo(&free, &total)) ;
//  std::cout << "GPU memory used(MB): " << ( total - free ) / 1000000
//            << "  bm table allocate size(MB), used:  " << CU_BigMem::bm_ptr->size() / 1000000 << ", "
//            << CU_BigMem::bm_ptr->used() / 1000000 << std::endl;
//  if ( (Rand4Hits *)rd4h ) delete (Rand4Hits *)rd4h  ;
//  if (CU_BigMem::bm_ptr)   delete CU_BigMem::bm_ptr  ;
//  
//  if (timing.count > 0) {
//    std::cout << "kernel timing\n";
//    printf("%12s %15s %15s\n","kernel","total /s","avg launch /s");
//    printf("%12s %15.8f %15.8f\n","sim_clean",timing.t_sim_clean.count(),
//           timing.t_sim_clean.count()/timing.count);
//    printf("%12s %15.8f %15.8f\n","sim_A",timing.t_sim_A.count(),
//           timing.t_sim_A.count()/timing.count);
//    printf("%12s %15.8f %15.8f\n","sim_ct",timing.t_sim_ct.count(),
//           timing.t_sim_ct.count()/timing.count);
//    printf("%12s %15.8f %15.8f\n","sim_cp",timing.t_sim_cp.count(),
//           timing.t_sim_cp.count()/timing.count);
//    printf("%12s %15d\n","launch count",timing.count);
//  } else {
//    std::cout << "no kernel timing available" << std::endl;
//  }
//  
//}
//
//
//__host__ void CaloGpuGeneral::load_hitsim_params(void * rd4h, HitParams* hp, long* simbins, int bins) {
// 
//    if( !(Rand4Hits *)rd4h ) { 
//	std::cout<<"Error load hit simulation params ! " ;
//	exit(2);
//	}
//
//    HitParams * hp_g = ((Rand4Hits *) rd4h )->get_hitparams() ;
//    long * simbins_g =  ((Rand4Hits *) rd4h) ->get_simbins() ;
//	
//   gpuQ(cudaMemcpy(hp_g, hp, bins*sizeof(HitParams), cudaMemcpyHostToDevice));
//   gpuQ(cudaMemcpy(simbins_g, simbins, bins*sizeof(long), cudaMemcpyHostToDevice));
//
//}
//
//
//
//__global__  void simulate_clean(Sim_Args args) {
// unsigned long  tid = threadIdx.x + blockIdx.x*blockDim.x ;
//  if ( tid < args.ncells * args.nsims ) { args.cells_energy[tid] = 0.0; }
// if(tid < args.nsims) args.ct[tid]= 0 ; 
//}
//
//__host__ int highestPowerof2( unsigned int n ) {
//    // Invalid input 
//  if ( n < 1 ) return 0;
//  
//    int res = 1; 
//  
//    // Try all powers starting from 2^1 
//  for ( unsigned int i = 0; i < 8 * sizeof( unsigned int ); i++ ) {
//        unsigned int curr = 1 << i; 
//  
//        // If current power is more than n, break 
//    if ( curr > n ) break;
//  
//        res = curr; 
//    } 
//  
//    return res; 
//}
//
//
//
//__device__  void CenterPositionCalculation_g_d(const HitParams hp, Hit& hit, const Sim_Args args) {
//
//  hit.setCenter_r( ( 1. - hp.extrapWeight ) * hp.extrapol_r_ent + hp.extrapWeight * hp.extrapol_r_ext );
//  hit.setCenter_z( ( 1. - hp.extrapWeight ) * hp.extrapol_z_ent + hp.extrapWeight * hp.extrapol_z_ext );
//  hit.setCenter_eta( ( 1. - hp.extrapWeight ) * hp.extrapol_eta_ent + hp.extrapWeight * hp.extrapol_eta_ext );
//  hit.setCenter_phi( ( 1. - hp.extrapWeight ) * hp.extrapol_phi_ent + hp.extrapWeight * hp.extrapol_phi_ext );
//}
//
//__device__ void HistoLateralShapeParametrization_g_d( const HitParams hp, Hit& hit, int t , Sim_Args args ) {
//
//  float  charge   = hp.charge;
//
//  float center_eta = hit.center_eta();
//  float center_phi = hit.center_phi();
//  float center_r   = hit.center_r();
//  float center_z   = hit.center_z();
//
//
//  float alpha, r, rnd1, rnd2;
//  rnd1 = args.rand[t];
//  rnd2 = args.rand[t+args.nhits];
//
//  if(hp.is_phi_symmetric) {
//    if(rnd2>=0.5) { //Fill negative phi half of shape
//      rnd2-=0.5;
//      rnd2*=2;
//      rnd_to_fct2d(alpha,r,rnd1,rnd2,hp.f2d);
//      alpha=-alpha;
//    } else { //Fill positive phi half of shape
//      rnd2*=2;
//      rnd_to_fct2d(alpha,r,rnd1,rnd2,hp.f2d);
//    }
//  } else {
//    rnd_to_fct2d(alpha,r,rnd1,rnd2, hp.f2d);
//  }
//
//
//  float delta_eta_mm = r * cos(alpha);
//  float delta_phi_mm = r * sin(alpha);
//
//  // Particles with negative eta are expected to have the same shape as those with positive eta after transformation:
//  // delta_eta --> -delta_eta
//  if(center_eta<0.)delta_eta_mm = -delta_eta_mm;
//  // Particle with negative charge are expected to have the same shape as positively charged particles after
//  // transformation: delta_phi --> -delta_phi
//  if(charge < 0.)  delta_phi_mm = -delta_phi_mm;
//
//  float dist000    = sqrt(center_r * center_r + center_z * center_z);
//  float eta_jakobi = abs(2.0 * exp(-center_eta) / (1.0 + exp(-2 * center_eta)));
//
//  float delta_eta = delta_eta_mm / eta_jakobi / dist000;
//  float delta_phi = delta_phi_mm / center_r;
//
//  hit.setEtaPhiZE(center_eta + delta_eta,center_phi + delta_phi,center_z, hit.E());
//
//
//}
//
//__device__ void HitCellMapping_g_d( HitParams hp,Hit& hit,  Sim_Args args ) {
//
// long long  cellele= getDDE(args.geo, hp.cs,hit.eta(),hit.phi());
//
////if (hp.index ==0 ) printf("Tid: %d cellId: %ld  nhits: %ld \n" , threadIdx.x ,cellele, hp.nhits ) ; 
//
// if( cellele < 0) printf("cellele not found %ld \n", cellele ) ; 
//  if( cellele >= 0 )  atomicAdd(&args.cells_energy[cellele+args.ncells*hp.index], hit.E()) ; 
//
//}
//
//__device__ void HitCellMappingWiggle_g_d( HitParams hp,Hit& hit, long t,  Sim_Args args  ) {
//
// FHs * f1d = hp.f1d ; 
// int nhist=(*f1d).nhist;
// float*  bin_low_edge = (*f1d ).low_edge ;
// 
// float eta =fabs( hit.eta()); 
//  if ( eta < bin_low_edge[0] || eta > bin_low_edge[nhist] ) { HitCellMapping_g_d( hp, hit, args ); }
//
// int bin= nhist ;
//  for (int i =0; i< nhist+1 ; ++i ) {
// 	if(bin_low_edge[i] > eta ) {
//	  bin = i ;
//	  break ;
//	}
//  }
//
////  bin=find_index_f(bin_low_edge, nhist+1, eta ) ;
//
//  bin -= 1; 
//
//  uint32_t * contents = (*f1d).h_contents[bin] ;
//  float* borders = (*f1d).h_borders[bin] ;
//  int h_size=(*f1d).h_szs[bin] ;
//  uint32_t s_MaxValue =(*f1d).s_MaxValue ;
//  
//
//     float rnd= args.rand[t+2*args.nhits];
//
//    float wiggle=rnd_to_fct1d(rnd,contents, borders, h_size, s_MaxValue);
//
//    float hit_phi_shifted=hit.phi()+wiggle;
//    hit.phi()=Phi_mpi_pi(hit_phi_shifted);
//  
//
////  HitCellMapping_g_d(hp, hit,  args) ;
//
//}
//
//
//
//__global__  void simulate_hits_de( const Sim_Args args ) {
//
//    long t = threadIdx.x + blockIdx.x*blockDim.x ;
//    if ( t  <  args.nhits ) {
//     Hit hit ;
//     int bin = find_index_long(args.simbins, args.nbins, t ) ;
////  if(bin == 0 ) printf("tid=%ld args.nbins=%d \n", t, args.nbins) ; 
//     HitParams hp =args.hitparams[bin] ;
//     hit.E()= hp.E ;
//     CenterPositionCalculation_g_d( hp, hit, args) ;
//     HistoLateralShapeParametrization_g_d(hp, hit, t, args) ;
//     if( hp.cmw)HitCellMappingWiggle_g_d ( hp, hit, t,args  ) ;
//     HitCellMapping_g_d(hp, hit, args) ;
//   }
//}
//
//__global__  void simulate_hits_ct( const Sim_Args args) {
//
// unsigned long tid = threadIdx.x + blockIdx.x*blockDim.x ;
// int sim = tid/args.ncells ; 
// unsigned long cellid=tid % args.ncells ;
// if( tid < args.ncells*args.nsims) {
//        if(args.cells_energy[tid] >0 )  {
//		unsigned int ct = atomicAdd(&args.ct[sim],1) ;
//		Cell_E ce;
//		ce.cellid=cellid ;
//		ce.energy=args.cells_energy[tid] ;
//		args.hitcells_E[ct + sim*MAXHITCT] = ce ;
////if(sim==0) printf("sim: %d  ct=%d cellid=%ld e=%f\n", sim, ct, cellid,  ce.energy); 
//	}
// }
//}
//
//
//__host__ void CaloGpuGeneral::simulate_hits_gr(Sim_Args &  args ) {
//
//  std::call_once(calledGetEnv, [](){
//        if(const char* env_p = std::getenv("FCS_BLOCK_SIZE")) {
//          std::string bs(env_p);
//          BLOCK_SIZE = std::stoi(bs);
//        }
//        if (BLOCK_SIZE != DEFAULT_BLOCK_SIZE) {
//          std::cout << "kernel BLOCK_SIZE: " << BLOCK_SIZE << std::endl;
//        }
//  });
//
//  // get Randowm numbers ptr , generate if need
//  long nhits =args.nhits ;
//  Rand4Hits * rd4h = (Rand4Hits *) args.rd4h ;
//  float * r= rd4h->rand_ptr(nhits)  ;
//  rd4h->add_a_hits(nhits) ;
//  args.rand =r;
//  
//  args.cells_energy =  rd4h->get_cells_energy() ;
//  args.hitcells_E = rd4h->get_cell_e() ;
//  args.hitcells_E_h = rd4h->get_cell_e_h() ;
//  args.ct = rd4h->get_ct() ;
//  args.ct_h = rd4h->get_ct_h() ;
//  
//  args.simbins=rd4h->get_simbins();
//  args.hitparams = rd4h->get_hitparams() ;
//    
//  cudaError_t err = cudaGetLastError();
//  
//  // clean up  for results ct[MAX_SIM] and hitcells_E[MAX_SIM*MAXHITCT]
//  // and workspace hitcells_energy[ncells*MAX_SIM]
//  
//  int blocksize=BLOCK_SIZE ;
//  int threads_tot= args.ncells*args.nsims  ;
//  int nblocks= (threads_tot + blocksize-1 )/blocksize ;
//  auto t0 = std::chrono::system_clock::now();
//  simulate_clean <<< nblocks, blocksize >>>( args) ;
//  gpuQ( cudaGetLastError() );
//  gpuQ( cudaDeviceSynchronize() );
//  
//  // Now main hit simulation find cell and populate hitcells_energy[] :
//  blocksize=BLOCK_SIZE ;
//  threads_tot= args.nhits  ;
//  nblocks= (threads_tot + blocksize-1 )/blocksize ;
//  auto t1 = std::chrono::system_clock::now();
//  simulate_hits_de <<<nblocks, blocksize >>> (args ) ;
//  gpuQ( cudaGetLastError() );
//  gpuQ( cudaDeviceSynchronize() );
//  
//  // Get result ct[] and hitcells_E[] (list of hitcells_ids/enengy )  
//  
//  nblocks = (args.ncells*args.nsims + blocksize -1 )/blocksize ;
//  auto t2 = std::chrono::system_clock::now();
//  simulate_hits_ct <<<nblocks, blocksize >>>(args ) ; 
//  gpuQ( cudaGetLastError() );
//  gpuQ( cudaDeviceSynchronize() );
//  
//  // cpy result back 
//  
//  auto t3 = std::chrono::system_clock::now();
//  gpuQ(cudaMemcpy(args.ct_h, args.ct, args.nsims*sizeof(int), cudaMemcpyDeviceToHost));
//  
//  gpuQ(
//      cudaMemcpy( args.hitcells_E_h, args.hitcells_E, MAXHITCT * MAX_SIM * sizeof( Cell_E ), cudaMemcpyDeviceToHost ) );
//  auto t4 = std::chrono::system_clock::now();
//  
//  CaloGpuGeneral::KernelTime kt(t1-t0, t2-t1, t3-t2, t4-t3);
//  timing += kt;
//  
//  //   for( int isim=0 ; isim<args.nsims ; isim++ ) 
//  //     gpuQ(cudaMemcpy(&args.hitcells_E_h[isim*MAXHITCT], &args.hitcells_E[isim*MAXHITCT],
//  //     args.ct_h[isim]*sizeof(Cell_E), cudaMemcpyDeviceToHost));
//} 
