/*
  Copyright (C) 2002-2021 CERN for the benefit of the ATLAS collaboration
*/

#include "GeoLoadGpu.h"

bool GeoLoadGpu::LoadGpu() {

#ifdef USE_KOKKOS
    return LoadGpu_kk();
#elif defined USE_STDPAR
    return LoadGpu_sp();
#else
    return LoadGpu_cu();
#endif
}
