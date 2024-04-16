/*
  Copyright (C) 2002-2021 CERN for the benefit of the ATLAS collaboration
*/

#ifndef CALOGPUGENERAL_STDPAR_H
#define CALOGPUGENERAL_STDPAR_H

#include "Args.h"

namespace CaloGpuGeneral_stdpar {

  void simulate_hits( float, int, Chain0_Args& );
  void Rand4Hits_finish( void* );  
  
} // namespace CaloGpuGeneral_stdpar
#endif
