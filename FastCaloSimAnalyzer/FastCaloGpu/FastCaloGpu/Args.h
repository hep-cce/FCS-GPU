#ifndef GPUARGS_H
#define GPUARGS_H

#include "FH_structs.h"

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
unsigned long long  seed ;


FH2D*  fh2d ;
FHs*   fhs ;



} Chain0_Args ;

#endif
