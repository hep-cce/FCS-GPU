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
//Hit * hits ;
bool * hitcells_b ;
unsigned long * hitcells ;
unsigned long * hitcells_l ;
unsigned int * hitcells_ct ;
unsigned long ncells ;


} Chain0_Args ;

#endif
