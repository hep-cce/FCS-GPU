

__device__ void  rnd_to_fct2d_sh(float& valuex,float& valuey,float rnd0,float rnd1,  FH2D hf2d_v, float*  borders) {

int nbinsx=hf2d_v.nbinsx;
int nbinsy=hf2d_v.nbinsy;


float *  bordersy= &borders[nbinsx+1] ;   //sharemem
float * contents= &bordersy[nbinsy+1] ;     //sharemem
/*
 int ibin = nbinsx*nbinsy-1 ;
 for ( int i=0 ; i < nbinsx*nbinsy ; ++i) {
    if   (contents[i]> rnd0 ) {
         ibin = i ;
         break ;
        }
 }
*/
int ibin=find_index_f(contents, nbinsx*nbinsy, rnd0 ) ;

  int biny = ibin/nbinsx;
  int binx = ibin - nbinsx*biny;

  float basecont=0;
  if(ibin>0) basecont=contents[ibin-1];

  float dcont=contents[ibin]-basecont;
  if(dcont>0) {
    valuex = borders[binx] + (borders[binx+1]-borders[binx]) * (rnd0-basecont) / dcont;
  } else {
    valuex = borders[binx] + (borders[binx+1]-borders[binx]) / 2;
  }
  valuey = bordersy[biny] + (bordersy[biny+1]-bordersy[biny]) * rnd1;

//if (threadIdx.x==0  ) 
//printf("blockIdx=%d,ibin=%d,rnd1=%f, rnd2=%f, nx=%d, ny=%d, valuex=%f, valuey%f \n",blockIdx.x, ibin, rnd0, rnd1, nbinsx, nbinsy,valuex,valuey );
//if (threadIdx.x==0 && blockIdx.x==2 ) {
//for (int ii=0;ii<nbinsx+1; ii++) printf("borderx[%d]=%f %f\n", ii, borders[ii],hf2d_v.h_bordersx[ii]);
//for (int ii=0;ii<nbinsy+1; ii++) printf("bordery[%d]=%f %f\n", ii, bordersy[ii],hf2d_v.h_bordersy[ii]);
//for (int ii=0; ii<nbinsx*nbinsy ; ii++) printf("Contents[%d]= %f %f\n",ii, contents[ii],HistoContents[ii]) ;
//}

}


__device__ void HistoLateralShapeParametrization_d_sh( Hit& hit, unsigned long t, Chain0_Args args, float*  borders ) {

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
      rnd_to_fct2d_sh(alpha,r,rnd1,rnd2,args.fh2d_v, borders);
      alpha=-alpha;
    } else { //Fill positive phi half of shape
      rnd2*=2;
      rnd_to_fct2d_sh(alpha,r,rnd1,rnd2,args.fh2d_v, borders);
    }
  } else {
    rnd_to_fct2d_sh(alpha,r,rnd1,rnd2,args.fh2d_v, borders);
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



 __global__  void simulate_chain0_A_sh( float E, int nhits,  Chain0_Args args ) {

extern __shared__ float borders[] ;

int nbinsx=args.fh2d_v.nbinsx;
int nbinsy=args.fh2d_v.nbinsy;


float *  bordersy= &borders[nbinsx+1] ;   //sharemem
float * contents= &bordersy[nbinsy+1] ;     //sharemem

float * HistoContents= args.fh2d_v.h_contents ;  //dev_mem
float* HistoBorders= args.fh2d_v.h_bordersx ;    //dev_mem
float* HistoBordersy= args.fh2d_v.h_bordersy ;    //dev_mem
// copy 2d funtion to shared mem.


 int index=threadIdx.x;
        
        while (index < nbinsx*nbinsy ){
                contents[index]=HistoContents[index] ;
        index+=blockDim.x ;
        }

 index= threadIdx.x ;
  while(index < (nbinsx+1)) {
        borders[index]=HistoBorders[index] ;
        index += blockDim.x ;
}

 index= threadIdx.x ;
  while(index < (nbinsy+1)) {
        bordersy[index]=HistoBordersy[index] ;
        index += blockDim.x ;
}
        __syncthreads() ;



// Now start:

   int tid = threadIdx.x + blockIdx.x*blockDim.x ;
   if ( tid  < nhits ){ 
    Hit hit ;
    hit.E()=E ;
    CenterPositionCalculation_d( hit, args) ;
    long long pre_cell = -5 ;
    HistoLateralShapeParametrization_d_sh(hit,tid,  args, borders ) ;
    if(args.spy) ValidationHitSpy_d( hit,  args, tid, pre_cell, false ) ;
    HitCellMappingWiggle_d ( hit, args, tid ) ;
    if(args.spy) ValidationHitSpy_d( hit,  args, tid , pre_cell,  true ) ;
   }
}
