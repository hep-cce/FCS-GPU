#include "Rand4Hits.h"

/*
//this is not used 
float *  Rand4Hits::HitsRandGen(unsigned int nhits, unsigned long long seed ) {

  gpuQ(cudaMalloc((void**)&m_rand_ptr , 3*nhits*sizeof(float))) ;
  CURAND_CALL(curandCreateGenerator(&m_gen, 
                CURAND_RNG_PSEUDO_DEFAULT));
  CURAND_CALL(curandSetPseudoRandomGeneratorSeed(m_gen, seed)) ;

  CURAND_CALL(curandGenerateUniform(m_gen, m_rand_ptr, 3*nhits));

   return m_rand_ptr ;
} 


*/
void  Rand4Hits::allocate_simulation(long long maxhits, unsigned short maxbins, unsigned short maxhitct, unsigned long n_cells){

float * Cells_Energy ;
gpuQ(cudaMalloc((void**)&Cells_Energy , n_cells* sizeof(float))) ;
m_cells_energy = Cells_Energy ;
Cell_E * cell_e ;
gpuQ(cudaMalloc((void**)&cell_e ,maxhitct* sizeof(Cell_E))) ;
m_cell_e = cell_e ; 
m_cell_e_h = (Cell_E * ) malloc(maxhitct* sizeof(Cell_E)) ; 
}

void  Rand4Hits::allocate_hist(long long maxhits, unsigned short maxbins, unsigned short maxhitct, int n_hist, int n_match, bool hitspy){
int n_float= 1+ (n_hist-n_match)+2*n_hist+1 ;   //7
int n_short= n_hist-1 ;        //2
int n_int= n_hist;    //3
int n_bool = 1+ n_match ;  //2
int n_ulong= 3 ;   
int n_uint=2 ;
int n_ull=1 ;
int n_double=n_hist*2+1 ;
 
m_F_ptrs = (float**)malloc(n_float*sizeof(float*)) ;
m_D_ptrs = (double**)malloc(n_double*sizeof(double*)) ;
m_S_ptrs = (short**)malloc(n_short*sizeof(short*)) ;
m_I_ptrs = (int**)malloc(n_int*sizeof(int*)) ;
m_B_ptrs= (bool**)malloc(n_bool*sizeof(bool*)) ;
m_Ul_ptrs= (unsigned long **) malloc(n_ulong*sizeof(unsigned long) );
m_Ui_ptrs =(unsigned int **) malloc(n_uint*sizeof(unsigned int ) ) ; 
m_Ull_ptrs= (unsigned long long **) malloc(n_ull*sizeof(unsigned long long) );

// for cellmapingwiggle ...
gpuQ(cudaMalloc((void**)&(m_B_ptrs[0]), 200000*sizeof(bool)   ));     //cells hits or not
gpuQ(cudaMalloc((void**)&(m_Ul_ptrs[0]), maxhits*sizeof(unsigned long)   )); //hit cells
gpuQ(cudaMalloc((void**)&(m_Ul_ptrs[1]), maxhitct*sizeof(unsigned long)   ));   // hit cells narrowed
gpuQ(cudaMalloc((void**)&(m_Ui_ptrs[0]), sizeof(unsigned int)   ));  //ct
gpuQ(cudaMalloc((void**)&(m_Ui_ptrs[1]), maxhitct*1024*sizeof(unsigned int)   )); //ct block inter

// for hitspy hists n_hists with n_matched 
for (int i =0; i<n_hist-n_match ; i++) {   //0 ,1
 gpuQ(cudaMalloc((void**)&(m_F_ptrs[i]), maxhits*sizeof(float)   ));   //hs x
 gpuQ(cudaMalloc((void**)&(m_S_ptrs[i]), maxhits*sizeof(short)   ));   //hs i
}

for (int i =0; i<n_hist; i++) 
 gpuQ(cudaMalloc((void**)&(m_I_ptrs[i]), 1024*maxbins*sizeof(int)   ));   //hs interm count

for (int i =0; i<n_match ; i++) {  
 gpuQ(cudaMalloc((void**)&(m_B_ptrs[n_match]), maxhits*sizeof(bool)   ));     //cells match or not
}


gpuQ(cudaMalloc((void**)&(m_F_ptrs[n_hist-n_match]), 1024*2*n_hist*sizeof(float)   ));   //[2]sumx, sumx2 interm

/*
for (int i =1+ (n_hist-n_match); i< 1+(n_hist-n_match)+n_hist; i++)    // 3,4,5
 gpuQ(cudaMalloc((void**)&(m_F_ptrs[i]), maxbins*sizeof(float)   ));   //hs array, cross event store.

for (int i= 1+(n_hist-n_match)+n_hist ; i < n_float-1; i++)  //6,7,8
 gpuQ(cudaMalloc((void**)&(m_F_ptrs[i]), maxbins*sizeof(float)   ));   //sumw2 array (w*array), cross event store.




gpuQ(cudaMalloc((void**)&(m_F_ptrs[n_float-1]), 4*n_hist*sizeof(float)   )); //[9]sumwx, sumwx2 cross event
*/

for ( int i=0; i<2*n_hist ; ++i) 
 gpuQ(cudaMalloc((void**)&(m_D_ptrs[i]), maxbins*sizeof(double)   ));   //hs main array and sumw2 cross event store.

 gpuQ(cudaMalloc((void**)&(m_D_ptrs[2*n_hist]), 4*n_hist*sizeof(double)   ));  //sumw, sumw2, sumwx, sumwx2 cross event store for each Hist





gpuQ(cudaMalloc((void**)&(m_Ull_ptrs[0]), 3*sizeof(unsigned long long )   )); //for store nentries


//host malloc.
m_hitcells =  (unsigned long * )malloc(maxhitct*sizeof(unsigned long) ) ;
m_hitcells_ct = (int*)malloc(maxhitct*sizeof(int)) ;


m_hist_stat_h =(double*) malloc(4* n_hist*sizeof(double)) ;

m_array_h_ptrs = (double**) malloc(n_hist*sizeof(double*) );
m_sumw2_array_h_ptrs = (double**) malloc(n_hist*sizeof(double*) );

for (int ii=0; ii<n_hist; ++ii ) {
m_array_h_ptrs[ii]=(double*) malloc(maxbins*sizeof(double)) ;
m_sumw2_array_h_ptrs[ii]=(double*) malloc(maxbins*sizeof(double)) ;
}

}


