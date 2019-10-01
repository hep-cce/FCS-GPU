__global__ void simulate_chain0_hit_ct_small(unsigned int * hitct_b, unsigned int ct , Chain0_Args args) {
extern __shared__ unsigned long hitcells[] ;
unsigned int *  counts = (unsigned int * ) (& hitcells[ct]) ;
 unsigned long  tid = threadIdx.x + blockIdx.x*blockDim.x ;
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

//      __syncthreads() ;
//if(tid==0) printf("index=%d, count=%d\n", iii, counts[iii] );
//if(threadIdx.x==0 ) for( int ii=0 ; ii<2 ; ii++) {printf("from Block-kernel block %d  counts[%d]=%d\n",blockIdx.x ,ii, counts[ii] ) ;}

__syncthreads() ;


for(int j =0 ; j< (ct+blockDim.x -1)/blockDim.x ; ++j ) {
        int index=threadIdx.x+j*blockDim.x;
        if(index < ct )
// now atomic update result
	atomicAdd(&(hitct_b[index]), counts[index] ) ;
        __syncthreads() ;
}


}

