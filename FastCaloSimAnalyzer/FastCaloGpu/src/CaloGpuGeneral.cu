#include "CaloGpuGeneral.h"
#include "GeoRegion.cu"
#include "Hit.h"

#include "Args.h"
#include "gpuQ.h"
#include <chrono>

#define BLOCK_SIZE 256

#define M_PI 3.14159265358979323846
#define M_2PI 6.28318530717958647692

__device__ long long getDDE( GeoGpu* geo, int sampling, float eta, float phi ) {

  float* distance = 0;
  int*   steps    = 0;

  int              MAX_SAMPLING = geo->max_sample;
  Rg_Sample_Index* SampleIdx    = geo->sample_index;
  GeoRegion*       regions_g    = geo->regions;

  if ( sampling < 0 ) return -1;
  if ( sampling >= MAX_SAMPLING ) return -1;

  int sample_size  = SampleIdx[sampling].size;
  int sample_index = SampleIdx[sampling].index;

  GeoRegion* gr = (GeoRegion*)regions_g;
  if ( sample_size == 0 ) return -1;
  float     dist;
  long long bestDDE = -1;
  if ( !distance ) distance = &dist;
  *distance = +10000000;
  int intsteps;
  int beststeps;
  if ( steps )
    beststeps = ( *steps );
  else
    beststeps = 0;

  if ( sampling < 21 ) {
    for ( int skip_range_check = 0; skip_range_check <= 1; ++skip_range_check ) {
      for ( unsigned int j = sample_index; j < sample_index + sample_size; ++j ) {
        if ( !skip_range_check ) {
          if ( eta < gr[j].mineta() ) continue;
          if ( eta > gr[j].maxeta() ) continue;
        }
        if ( steps )
          intsteps = ( *steps );
        else
          intsteps = 0;
        float     newdist;
        long long newDDE = gr[j].getDDE( eta, phi, &newdist, &intsteps );
        if ( newdist < *distance ) {
          bestDDE   = newDDE;
          *distance = newdist;
          if ( steps ) beststeps = intsteps;
          if ( newdist < -0.1 ) break; // stop, we are well within the hit cell
        }
      }
      if ( bestDDE >= 0 ) break;
    }
  } else {
    return -3;
  }
  if ( steps ) *steps = beststeps;

  return bestDDE;
}

__device__ int find_index_f( float* array, int size, float value ) {
  // fist index (from 0)  have element value > value
  // array[i] > value ; array[i-1] <= value
  // std::upbund( )

  int low     = 0;
  int high    = size - 1;
  int m_index = ( high - low ) / 2;
  while ( high != low ) {
    if ( value >= array[m_index] )
      low = m_index + 1;
    else
      high = m_index;
    m_index = ( high + low ) / 2;
  }
  return m_index;
}

__device__ int find_index_uint32( uint32_t* array, int size, float value ) {
  // find the first index of element which has vaule > value
  int low     = 0;
  int high    = size - 1;
  int m_index = ( high - low ) / 2;
  while ( high != low ) {
    if ( value > array[m_index] )
      low = m_index + 1;
    else if ( value == array[m_index] ) {
      return m_index + 1;
    } else
      high = m_index;
    m_index = ( high - low ) / 2 + low;
  }
  return m_index;
}

__device__ void rnd_to_fct2d( float& valuex, float& valuey, float rnd0, float rnd1, FH2D* hf2d ) {

  int    nbinsx        = ( *hf2d ).nbinsx;
  int    nbinsy        = ( *hf2d ).nbinsy;
  float* HistoContents = ( *hf2d ).h_contents;
  float* HistoBorders  = ( *hf2d ).h_bordersx;
  float* HistoBordersy = ( *hf2d ).h_bordersy;

  /*
   int ibin = nbinsx*nbinsy-1 ;
   for ( int i=0 ; i < nbinsx*nbinsy ; ++i) {
      if   (HistoContents[i]> rnd0 ) {
           ibin = i ;
           break ;
          }
   }
  */
  int ibin = find_index_f( HistoContents, nbinsx * nbinsy, rnd0 );

  int biny = ibin / nbinsx;
  int binx = ibin - nbinsx * biny;

  float basecont = 0;
  if ( ibin > 0 ) basecont = HistoContents[ibin - 1];

  float dcont = HistoContents[ibin] - basecont;
  if ( dcont > 0 ) {
    valuex = HistoBorders[binx] + ( HistoBorders[binx + 1] - HistoBorders[binx] ) * ( rnd0 - basecont ) / dcont;
  } else {
    valuex = HistoBorders[binx] + ( HistoBorders[binx + 1] - HistoBorders[binx] ) / 2;
  }
  valuey = HistoBordersy[biny] + ( HistoBordersy[biny + 1] - HistoBordersy[biny] ) * rnd1;
}

__device__ float rnd_to_fct1d( float rnd, uint32_t* contents, float* borders, int nbins, uint32_t s_MaxValue ) {

  uint32_t int_rnd = s_MaxValue * rnd;
  /*
    int  ibin=nbins-1 ;
    for ( int i=0 ; i < nbins ; ++i) {
      if   (contents[i]> int_rnd ) {
           ibin = i ;
           break ;
          }
    }
  */
  int ibin = find_index_uint32( contents, nbins, int_rnd );

  int binx = ibin;

  uint32_t basecont = 0;
  if ( ibin > 0 ) basecont = contents[ibin - 1];

  uint32_t dcont = contents[ibin] - basecont;
  if ( dcont > 0 ) {
    return borders[binx] + ( ( borders[binx + 1] - borders[binx] ) * ( int_rnd - basecont ) ) / dcont;
  } else {
    return borders[binx] + ( borders[binx + 1] - borders[binx] ) / 2;
  }
}

__device__ void CenterPositionCalculation_d( Hit& hit, const Chain0_Args args ) {

  hit.setCenter_r( ( 1. - args.extrapWeight ) * args.extrapol_r_ent + args.extrapWeight * args.extrapol_r_ext );
  hit.setCenter_z( ( 1. - args.extrapWeight ) * args.extrapol_z_ent + args.extrapWeight * args.extrapol_z_ext );
  hit.setCenter_eta( ( 1. - args.extrapWeight ) * args.extrapol_eta_ent + args.extrapWeight * args.extrapol_eta_ext );
  hit.setCenter_phi( ( 1. - args.extrapWeight ) * args.extrapol_phi_ent + args.extrapWeight * args.extrapol_phi_ext );
}

__device__ void HistoLateralShapeParametrization_d( Hit& hit, unsigned long t, Chain0_Args args ) {

  // int     pdgId    = args.pdgId;
  float charge = args.charge;

  // int cs=args.charge;
  float center_eta = hit.center_eta();
  float center_phi = hit.center_phi();
  float center_r   = hit.center_r();
  float center_z   = hit.center_z();

  float alpha, r, rnd1, rnd2;
  rnd1 = args.rand[t];
  rnd2 = args.rand[t + args.nhits];

  if ( args.is_phi_symmetric ) {
    if ( rnd2 >= 0.5 ) { // Fill negative phi half of shape
      rnd2 -= 0.5;
      rnd2 *= 2;
      rnd_to_fct2d( alpha, r, rnd1, rnd2, args.fh2d );
      alpha = -alpha;
    } else { // Fill positive phi half of shape
      rnd2 *= 2;
      rnd_to_fct2d( alpha, r, rnd1, rnd2, args.fh2d );
    }
  } else {
    rnd_to_fct2d( alpha, r, rnd1, rnd2, args.fh2d );
  }

  float delta_eta_mm = r * cos( alpha );
  float delta_phi_mm = r * sin( alpha );

  // Particles with negative eta are expected to have the same shape as those with positive eta after transformation:
  // delta_eta --> -delta_eta
  if ( center_eta < 0. ) delta_eta_mm = -delta_eta_mm;
  // Particle with negative charge are expected to have the same shape as positively charged particles after
  // transformation: delta_phi --> -delta_phi
  if ( charge < 0. ) delta_phi_mm = -delta_phi_mm;

  float dist000    = sqrt( center_r * center_r + center_z * center_z );
  float eta_jakobi = abs( 2.0 * exp( -center_eta ) / ( 1.0 + exp( -2 * center_eta ) ) );

  float delta_eta = delta_eta_mm / eta_jakobi / dist000;
  float delta_phi = delta_phi_mm / center_r;

  hit.setEtaPhiZE( center_eta + delta_eta, center_phi + delta_phi, center_z, hit.E() );
}

__device__ void HitCellMapping_d( Hit& hit, unsigned long t, Chain0_Args args ) {

  long long cellele = getDDE( args.geo, args.cs, hit.eta(), hit.phi() );

  if ( cellele < 0 ) printf( "cellele not found %ld \n", cellele );

  //  args.hitcells_b[cellele]= true ;
  //  args.hitcells[t]=cellele ;

  atomicAdd( &args.cells_energy[cellele], hit.E() );

  /*
    CaloDetDescrElement cell =( *(args.geo)).cells[cellele] ;
    long long id = cell.identify();
    float eta=cell.eta();
    float phi=cell.phi();
    float z=cell.z();
    float r=cell.r() ;
  */
}

__device__ void HitCellMappingWiggle_d( Hit& hit, Chain0_Args args, unsigned long t ) {

  int    nhist        = ( *( args.fhs ) ).nhist;
  float* bin_low_edge = ( *( args.fhs ) ).low_edge;

  float eta = fabs( hit.eta() );
  if ( eta < bin_low_edge[0] || eta > bin_low_edge[nhist] ) { HitCellMapping_d( hit, t, args ); }

  int bin = nhist;
  for ( int i = 0; i < nhist + 1; ++i ) {
    if ( bin_low_edge[i] > eta ) {
      bin = i;
      break;
    }
  }

  //  bin=find_index_f(bin_low_edge, nhist+1, eta ) ;

  bin -= 1;

  uint32_t* contents   = ( *( args.fhs ) ).h_contents[bin];
  float*    borders    = ( *( args.fhs ) ).h_borders[bin];
  int       h_size     = ( *( args.fhs ) ).h_szs[bin];
  uint32_t  s_MaxValue = ( *( args.fhs ) ).s_MaxValue;

  float rnd = args.rand[t + 2 * args.nhits];

  float wiggle = rnd_to_fct1d( rnd, contents, borders, h_size, s_MaxValue );

  float hit_phi_shifted = hit.phi() + wiggle;
  hit.phi()             = Phi_mpi_pi( hit_phi_shifted );

  HitCellMapping_d( hit, t, args );
}

__global__ void simulate_A( float E, int nhits, Chain0_Args args ) {

  long t = threadIdx.x + blockIdx.x * blockDim.x;
  if ( t < nhits ) {
    Hit hit;
    hit.E() = E;
    CenterPositionCalculation_d( hit, args );
    HistoLateralShapeParametrization_d( hit, t, args );
    HitCellMappingWiggle_d( hit, args, t );
  }
}

__global__ void simulate_ct( Chain0_Args args ) {

  unsigned long tid = threadIdx.x + blockIdx.x * blockDim.x;
  if ( tid < args.ncells ) {
    if ( args.cells_energy[tid] > 0 ) {
      unsigned int ct = atomicAdd( args.hitcells_ct, 1 );
      Cell_E       ce;
      ce.cellid           = tid;
      ce.energy           = args.cells_energy[tid];
      args.hitcells_E[ct] = ce;
    }
  }
}

__host__ void* CaloGpuGeneral::Rand4Hits_init( long long maxhits, unsigned short maxbin, unsigned long long seed,
                                               bool hitspy ) {

  auto              t0   = std::chrono::system_clock::now();
  Rand4Hits*        rd4h = new Rand4Hits;
  float*            f;
  curandGenerator_t gen;
  auto              t1 = std::chrono::system_clock::now();

  CURAND_CALL( curandCreateGenerator( &gen, CURAND_RNG_PSEUDO_DEFAULT ) );
  CURAND_CALL( curandSetPseudoRandomGeneratorSeed( gen, seed ) );
  auto t2 = std::chrono::system_clock::now();
  gpuQ( cudaMalloc( (void**)&f, 3 * maxhits * sizeof( float ) ) );
  auto t3 = std::chrono::system_clock::now();
  rd4h->set_rand_ptr( f );
  rd4h->set_gen( gen );
  rd4h->set_t_a_hits( maxhits );
  rd4h->set_c_hits( 0 );
  CURAND_CALL( curandGenerateUniform( gen, f, 3 * maxhits ) );
  auto t4 = std::chrono::system_clock::now();

  rd4h->allocate_simulation( maxhits, maxbin, 2000, 200000 );
  auto t5 = std::chrono::system_clock::now();

  std::chrono::duration<double> diff1 = t1 - t0;
  std::chrono::duration<double> diff2 = t2 - t1;
  std::chrono::duration<double> diff3 = t3 - t2;
  std::chrono::duration<double> diff4 = t4 - t3;
  std::chrono::duration<double> diff5 = t5 - t4;
  std::cout << "Time of R4hit: " << diff1.count() << "," << diff2.count() << "," << diff3.count() << ","
            << diff4.count() << "," << diff5.count() << " s" << std::endl;

  return (void*)rd4h;
}

__host__ void CaloGpuGeneral::Rand4Hits_finish( void* rd4h ) {

  if ( (Rand4Hits*)rd4h ) delete (Rand4Hits*)rd4h;
}

__global__ void simulate_clean( Chain0_Args args ) {
  unsigned long tid = threadIdx.x + blockIdx.x * blockDim.x;
  if ( tid < args.ncells ) { args.cells_energy[tid] = 0.0; }
  if ( tid == 0 ) args.hitcells_ct[0] = 0;
}

__host__ void CaloGpuGeneral::simulate_hits( float E, int nhits, Chain0_Args& args ) {

  Rand4Hits* rd4h = (Rand4Hits*)args.rd4h;

  float* r = rd4h->rand_ptr( nhits );

  rd4h->add_a_hits( nhits );
  args.rand = r;

  unsigned long ncells = args.ncells;
  args.maxhitct        = MAXHITCT;

  args.cells_energy = rd4h->get_cells_energy(); // Hit cell energy map , size of ncells(~200k float)
  args.hitcells_E   = rd4h->get_cell_e();       // Hit cell energy map, moved together
  args.hitcells_E_h = rd4h->get_cell_e_h();     // Host array

  args.hitcells_ct = rd4h->get_ct(); // single value, number of  uniq hit cells

  cudaError_t err = cudaGetLastError();

  int blocksize   = BLOCK_SIZE;
  int threads_tot = args.ncells;
  int nblocks     = ( threads_tot + blocksize - 1 ) / blocksize;

  simulate_clean<<<nblocks, blocksize>>>( args );
  // 	cudaDeviceSynchronize() ;
  // if (err != cudaSuccess) {
  //       std::cout<< "simulate_clean "<<cudaGetErrorString(err)<< std::endl;
  //}

  blocksize   = BLOCK_SIZE;
  threads_tot = nhits;
  nblocks     = ( threads_tot + blocksize - 1 ) / blocksize;

  //	 std::cout<<"Nblocks: "<< nblocks << ", blocksize: "<< blocksize
  //               << ", total Threads: " << threads_tot << std::endl ;

  //  int fh_size=args.fh2d_v.nbinsx+args.fh2d_v.nbinsy+2+(args.fh2d_v.nbinsx+1)*(args.fh2d_v.nbinsy+1) ;
  // if(args.debug) std::cout<<"2DHisto_Func_size: " << args.fh2d_v.nbinsx << ", " << args.fh2d_v.nbinsy << "= " <<
  // fh_size <<std::endl ;

  simulate_A<<<nblocks, blocksize>>>( E, nhits, args );

  //  cudaDeviceSynchronize() ;
  //  err = cudaGetLastError();
  // if (err != cudaSuccess) {
  //        std::cout<< "simulate_A "<<cudaGetErrorString(err)<< std::endl;
  //}

  nblocks = ( ncells + blocksize - 1 ) / blocksize;
  simulate_ct<<<nblocks, blocksize>>>( args );
  //  cudaDeviceSynchronize() ;
  // err = cudaGetLastError();
  // if (err != cudaSuccess) {
  //        std::cout<< "simulate_chain0_B1 "<<cudaGetErrorString(err)<< std::endl;
  //}

  int ct;
  gpuQ( cudaMemcpy( &ct, args.hitcells_ct, sizeof( int ), cudaMemcpyDeviceToHost ) );
  // std::cout<< "ct="<<ct<<std::endl;
  gpuQ( cudaMemcpy( args.hitcells_E_h, args.hitcells_E, ct * sizeof( Cell_E ), cudaMemcpyDeviceToHost ) );

  // pass result back
  args.ct = ct;
  //   args.hitcells_ct_h=hitcells_ct ;
}
