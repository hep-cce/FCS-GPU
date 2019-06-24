#ifndef GPUARGS_H
#define GPUARGS_H

#include "FH_structs.h"
#include "GeoGpu_structs.h"
#include "Hit.h"

#define MAXHITS 200000 

typedef struct Chain0_Args {

float extrapol_eta_ent ;
float extrapol_phi_ent ;
float extrapol_r_ent ;
float extrapol_z_ent ;
float extrapol_eta_ext ;
float extrapol_phi_ext ;
float extrapol_r_ext ;
float extrapol_z_ext ;

float extrapWeight ;

int pdgId ;
double charge ;
int cs ;
bool is_phi_symmetric ;
float * rand ;
int nhits ;
void * rd4h ;

FH2D*  fh2d ;
FHs*   fhs ;

GeoGpu * geo ;

bool * hitcells_b ;  // GPU array of whether a cell got hit
unsigned long * hitcells ;//GPU pointer for hit cell index for each hit
unsigned long * hitcells_l ; // GPU pointer for uniq  hitcell indexes  
unsigned int * hitcells_ct ;  //GPU pointer for array(ct*C1numBlocks) for accumulate hit counts
unsigned long ncells ;

unsigned long * hitcells_h ; //Host array of hit cell index
int * hitcells_ct_h ; // host array of corresponding hit cell counts
unsigned int ct ;  // cells got hit for the event

bool spy ;
bool isBarrel ; 

} Chain0_Args ;

#endif
