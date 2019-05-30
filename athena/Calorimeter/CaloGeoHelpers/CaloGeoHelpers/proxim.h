/*
  Copyright (C) 2002-2017 CERN for the benefit of the ATLAS collaboration
*/

#ifndef CALOGEOHELPER_PROXIM
#define CALOGEOHELPER_PROXIM

// inline function for handling two phi angles, to avoid 2PI wrap around. 
//

#include <math.h> 
#include "CxxUtils/AthUnlikelyMacros.h"

inline double proxim(double b,double a)
{
  const double aplus = a + M_PI;
  const double aminus = a - M_PI;
  if (ATH_UNLIKELY(b > aplus)) {
    do {
      b -= 2*M_PI;
    } while(b > aplus);
  }
  else if (ATH_UNLIKELY(b < aminus)) {
    do {
      b += 2*M_PI;
    } while(b < aminus);
  }
  return b;
}


#endif
