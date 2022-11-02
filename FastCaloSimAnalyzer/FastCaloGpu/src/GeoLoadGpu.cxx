/*
  Copyright (C) 2002-2021 CERN for the benefit of the ATLAS collaboration
*/

#include "GeoLoadGpu.h"

bool GeoLoadGpu::LoadGpu() {

#if defined (USE_STDPAR)
    return LoadGpu_sp();
#elif defined (USE_KOKKOS)
    return LoadGpu_kk();
#else
    return LoadGpu_cu();
#endif
}
