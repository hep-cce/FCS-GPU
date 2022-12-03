/*
  Copyright (C) 2002-2023 CERN for the benefit of the ATLAS collaboration
*/

#include "Rand4Hits.h"

void  Rand4Hits::allocate_simulation( int  maxbins, int  maxhitct, unsigned long n_cells){

float * Cells_Energy ;
gpuQ(hipMalloc((void**)&Cells_Energy , MAX_SIM * n_cells* sizeof(float))) ;
m_cells_energy = Cells_Energy ;

Cell_E * cell_e ;
gpuQ(hipMalloc((void**)&cell_e ,MAX_SIM*maxhitct* sizeof(Cell_E))) ;
m_cell_e = cell_e ; 
m_cell_e_h = (Cell_E * ) malloc(MAX_SIM*maxhitct* sizeof(Cell_E)) ; 

long * simbins ;
gpuQ(hipMalloc((void**)&simbins ,MAX_SIMBINS* sizeof(long))) ;
m_simbins = simbins ;

HitParams * hitparams ;
gpuQ(hipMalloc((void**)&hitparams ,MAX_SIMBINS* sizeof(HitParams))) ;
m_hitparams = hitparams ;

int *  ct_ptr ;
gpuQ(hipMalloc((void**)&ct_ptr ,MAX_SIM* sizeof(int))) ;
m_ct = ct_ptr ;
m_ct_h = (int* ) malloc(MAX_SIM*sizeof(int)) ;

}

