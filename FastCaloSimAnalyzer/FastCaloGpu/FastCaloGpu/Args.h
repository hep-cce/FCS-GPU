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



struct Bin1 {
  double x;
  double fTsumw;
  double fTsumw2;
  double fTsumwx;
  double fTsumwx2;

  CUDA_HOSTDEV inline void set(double xp, double w) {
    x = xp;
    fTsumw = w;
    fTsumw2 = w*w;
    fTsumwx = w*x;
    fTsumwx2 = w*x*x;
  }
};

struct Bin2 {
  double x;
  double y;
  double fTsumw;
  double fTsumw2;
  double fTsumwx;
  double fTsumwx2;
  double fTsumwy;
  double fTsumwy2;
  double fTsumwxy;
 
  CUDA_HOSTDEV inline void set(double xp, double yp, double w) {
    x = xp;
    y = yp;
    fTsumw   = w;
    fTsumw2  = w*w;
    fTsumwx  = w*x;
    fTsumwx2 = w*x*x;
    fTsumwy  = w*y;
    fTsumwy2 = w*y*y;
    fTsumwxy = w*x*y; 
  }
};

struct Bins {
  Bin1 m_hist_hitgeo_dphi; 
  Bin1 m_hist_deltaEta;
  Bin1 m_hist_deltaPhi;
  Bin1 m_hist_deltaRt;
  Bin1 m_hist_deltaZ;

  Bin2 m_hist_hitenergy_alpha_radius;
  Bin2 m_hist_hitenergy_alpha_absPhi_radius;

  Bin1 m_hist_total_hitPhi_minus_cellPhi;
  Bin1 m_hist_matched_hitPhi_minus_cellPhi;

  Bin1 m_hist_total_hitPhi_minus_cellPhi_etaboundary;
  Bin1 m_hist_matched_hitPhi_minus_cellPhi_etaboundary;

  Bin2 m_hist_Rz;
  Bin2 m_hist_Rz_outOfRange;

  Bin1 m_hist_hitenergy_weight;
  Bin1 m_hist_hitenergy_r;
  Bin1 m_hist_hitenergy_z;
  Bin1 m_hist_hitgeo_matchprevious_dphi;

};



#endif
