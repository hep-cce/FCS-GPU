/*
  Copyright (C) 2002-2021 CERN for the benefit of the ATLAS collaboration
*/
//#include "gpuQ.h"
#include "Rand4Hits.h"
#include <omp.h>

void  Rand4Hits::allocate_simulation( int  maxbins, int  maxhitct, unsigned long n_cells){

int m_default_device = omp_get_default_device();
int m_initial_device = omp_get_initial_device();

float * Cells_Energy ;
//gpuQ(cudaMalloc((void**)&Cells_Energy , MAX_SIM * n_cells* sizeof(float))) ;
printf("aaaaaaaaaaaaaaa %d", MAX_SIM * n_cells);
Cells_Energy = (float *) omp_target_alloc( MAX_SIM * n_cells* sizeof(float), m_default_device);
m_cells_energy = Cells_Energy ;

Cell_E * cell_e ;
//gpuQ(cudaMalloc((void**)&cell_e ,MAX_SIM*maxhitct* sizeof(Cell_E))) ;
cell_e = (Cell_E *) omp_target_alloc( MAX_SIM*maxhitct* sizeof(Cell_E), m_default_device);
m_cell_e = cell_e ; 
m_cell_e_h = (Cell_E * ) malloc(MAX_SIM*maxhitct* sizeof(Cell_E)) ; 

long * simbins ;
//gpuQ(cudaMalloc((void**)&simbins ,MAX_SIMBINS* sizeof(long))) ;
simbins = (long *) omp_target_alloc( MAX_SIMBINS* sizeof(long), m_default_device);
m_simbins = simbins ;

HitParams * hitparams ;
//gpuQ(cudaMalloc((void**)&hitparams ,MAX_SIMBINS* sizeof(HitParams))) ;
hitparams = (HitParams *) omp_target_alloc( MAX_SIMBINS* sizeof(HitParams), m_default_device);
m_hitparams = hitparams ;

int *  ct_ptr ;
//gpuQ(cudaMalloc((void**)&ct_ptr ,MAX_SIM* sizeof(int))) ;
ct_ptr = (int *) omp_target_alloc( MAX_SIM* sizeof(int), m_default_device);
m_ct = ct_ptr ;
m_ct_h = (int* ) malloc(MAX_SIM*sizeof(int)) ;

}

