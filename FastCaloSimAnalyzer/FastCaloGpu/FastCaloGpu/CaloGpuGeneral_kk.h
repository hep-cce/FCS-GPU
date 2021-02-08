#ifndef CALOGPUGENERAL_kk_H
#define CALOGPUGENERAL_kk_H

#include "Args.h"

namespace CaloGpuGeneral_kk {

  void Rand4Hits_finish( void* rd4h );
  void simulate_hits( float E, int nhits, Chain0_Args& args );

} // namespace CaloGpuGeneral_kk
#endif
