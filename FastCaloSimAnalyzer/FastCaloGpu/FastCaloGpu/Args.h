#ifndef GPUARGS_H
#define GPUARGS_H

#include "FH_structs.h"
#include "GeoGpu_structs.h"
#include "Hit.h"

#include "Hitspy_Hist.h"

#define MAXHITS 200000 
#define MAXBINS 1024  
#define MAXHITCT 2000  

typedef struct Chain0_Args {

bool debug ;

float extrapol_eta_ent ;
float extrapol_phi_ent ;
float extrapol_r_ent ;
float extrapol_z_ent ;
float extrapol_eta_ext ;
float extrapol_phi_ext ;
float extrapol_r_ext ;
float extrapol_z_ext ;

float extrapWeight ;
float E ;

int pdgId ;
double charge ;
int cs ;
bool is_phi_symmetric ;
float * rand ;
int nhits ;
void * rd4h ;

FH2D*  fh2d ;
FHs*   fhs ;
FH2D	fh2d_v ;  // host struct so we have info

GeoGpu * geo ;

bool is_first ; // first event 
bool is_last ; // last event 

bool * hitcells_b ;  // GPU array of whether a cell got hit
unsigned long * hitcells ;//GPU pointer for hit cell index for each hit
unsigned long * hitcells_l ; // GPU pointer for uniq  hitcell indexes  
unsigned int * hitcells_ct ;  //GPU pointer for number of uniq hit cells 
unsigned long ncells ;
unsigned int maxhitct;

unsigned int * hitcounts_b ; // GPU pointer for interm blockwise result of hit counts

unsigned long * hitcells_h ; //Host array of hit cell index
int * hitcells_ct_h ; // host array of corresponding hit cell counts
unsigned int ct ;  // cells got hit for the event

bool spy ;
bool isBarrel ; 
Hitspy_Hist hs1 ;
Hitspy_Hist hs2 ;

float * hs_sumx ; // for staged sumx , sumx2  6*1024
double * hs_sumwx_g ; // for cross event sumwx on gpu 8 numbers
double * hs_sumwx_h ; // for host stat array ;

unsigned long long * hs_nentries ; //gpu point for 3* entry, accumulation of nhits

} Chain0_Args ;




#endif
