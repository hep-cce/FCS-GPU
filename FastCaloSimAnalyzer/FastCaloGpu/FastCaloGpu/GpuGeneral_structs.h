#ifndef GPU_GENERAL_STRUCT_H
#define GPU_GENERAL_STRUCT_H

#include "FH_structs.h"
typedef struct Cell_E {
  unsigned long cellid ;
  float energy ;
} Cell_E ;


typedef struct HitParams {
  int index ; //simulate index for gpu simulation
  int cs ;
  float E ; 
  int pdgId ;
  double charge ;
  bool is_phi_symmetric ;
  long nhits ;
  FHs * f1d ;
  FH2D * f2d ;   
  double extrapol_eta_ent; 
  double extrapol_eta_ext; 
  double extrapol_phi_ent; 
  double extrapol_phi_ext; 
  double extrapol_r_ent; 
  double extrapol_r_ext; 
  double extrapol_z_ent; 
  double extrapol_z_ext; 
  float extrapWeight ;
  bool cmw ; //Do CellMapingWiggle or direct CellMapping
} HitParams ;
#endif
