/*
  Copyright (C) 2002-2021 CERN for the benefit of the ATLAS collaboration
*/

#ifndef CALOGPUGENERAL_OMP_H
#define CALOGPUGENERAL_OMP_H

#include "Args.h"

namespace CaloGpuGeneral_omp {

void Rand4Hits_finish(void *rd4h);
void simulate_hits(float, int, Chain0_Args &, int);
void simulate_A_cu(float, int, Chain0_Args &);
void Rand4Hits_finish(void *);

} // namespace CaloGpuGeneral_omp
#endif
