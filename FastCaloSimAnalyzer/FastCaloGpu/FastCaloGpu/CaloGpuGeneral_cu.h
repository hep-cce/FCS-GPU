#ifndef CALOGPUGENERAL_CU_H
#define CALOGPUGENERAL_CU_H

#include "Args.h"

namespace CaloGpuGeneral_cu {

  void simulate_hits( float, int, Chain0_Args& );
  void simulate_A_cu( float, int, Chain0_Args& );

} // namespace CaloGpuGeneral_cu
#endif
