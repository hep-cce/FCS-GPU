#ifndef CALOGPUGENERAL_AL_H
#define CALOGPUGENERAL_AL_H

#include "Args.h"

namespace CaloGpuGeneral_al {
  void Rand4Hits_finish( void* rd4h );
  void simulate_hits_gr(Sim_Args &);
  void load_hitsim_params(void *, HitParams *, long *, int);
}

#endif
