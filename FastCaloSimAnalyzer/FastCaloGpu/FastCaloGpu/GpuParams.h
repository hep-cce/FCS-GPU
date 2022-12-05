/*
  Copyright (C) 2002-2021 CERN for the benefit of the ATLAS collaboration
*/

#ifndef GPU_PARAMS_H
#define GPU_PARAMS_H

#define MAXHITS 1500000
#define MAXBINS 1024
#define MAXHITCT 2000

#define MIN_GPU_HITS 256
#define MAX_SIM 300
#define MAX_SIMBINS MAX_SIM * 24

#define MAX_CELLS 200000

// Size of cuMalloc each time
#define M_SEG_SIZE 134217728
//#define M_SEG_SIZE  67108864
//#define M_SEG_SIZE  268435456

#endif
