/*
  Copyright (C) 2002-2017 CERN for the benefit of the ATLAS collaboration
*/

// $Id$
/**
 * @file  CaloPhiRange_test.cxx
 * @author scott snyder <snyder@bnl.gov>
 * @date Jul, 2013
 * @brief Component test for CaloPhiRange.
 */

#undef NDEBUG

#include "CaloGeoHelpers/CaloPhiRange.h"
#include <iostream>
#include <cassert>
#include <cmath>


void test1()
{
  std::cout << "test1\n";
  CaloPhiRange r;
  assert (r.twopi() == 2*M_PI);
  assert (r.phi_min() == -M_PI);
  assert (r.phi_max() ==  M_PI);

  assert (r.fix(0.3) == 0.3);
  assert (r.fix(0.3 + M_PI) == 0.3 - M_PI);
  assert (r.fix(-M_PI - 0.3) == M_PI - 0.3);

  assert (r.diff (0.5, 0.3) == 0.2);
  assert (std::abs(r.diff (M_PI - 0.1, -M_PI+0.1) + 0.2) < 1e-12);

  r.print();
}


void test2()
{
  std::cout << "test2\n";
  assert (CaloPhiRange::twopi() == 2*M_PI);
  assert (CaloPhiRange::phi_min() == -M_PI);
  assert (CaloPhiRange::phi_max() ==  M_PI);

  assert (CaloPhiRange::fix(0.3) == 0.3);
  assert (CaloPhiRange::fix(0.3 + M_PI) == 0.3 - M_PI);
  assert (CaloPhiRange::fix(-M_PI - 0.3) == M_PI - 0.3);

  assert (CaloPhiRange::diff (0.5, 0.3) == 0.2);
  assert (std::abs(CaloPhiRange::diff (M_PI - 0.1, -M_PI+0.1) + 0.2) < 1e-12);

  CaloPhiRange::print();
}


int main()
{
  test1();
  test2();
  return 0;
}
