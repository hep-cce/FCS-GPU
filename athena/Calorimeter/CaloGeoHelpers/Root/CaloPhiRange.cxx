/*
  Copyright (C) 2002-2017 CERN for the benefit of the ATLAS collaboration
*/

/***************************************************************************
 Liquid Argon detector description package
 -----------------------------------------
 Copyright (C) 1998 by ATLAS Collaboration
 ***************************************************************************/

//<doc><file>	$Id: CaloPhiRange.cxx 587585 2014-03-14 08:32:57Z krasznaa $
//<version>	$Name: not supported by cvs2svn $

#include "CaloGeoHelpers/CaloPhiRange.h"

#include <cmath>
#include <iostream>
//#include <iomanip>

const double CaloPhiRange::m_phi_min = -M_PI;
const double CaloPhiRange::m_twopi = 2*M_PI;
const double CaloPhiRange::m_phi_max = M_PI;


double
CaloPhiRange::fix ( double phi )
{
  if (phi < m_phi_min) return (phi+m_twopi);
  if (phi > m_phi_max) return (phi-m_twopi);
  return phi;
}

double
CaloPhiRange::diff ( double phi1, double phi2 )
{
  double res = fix(phi1) - fix(phi2);
  return fix(res);
}
void 			
CaloPhiRange::print		()
{
  
  std::cout << std::endl << " Phi Range used in calo's is " << m_phi_min 
	    << " to " << m_phi_max << std::endl << std::endl;
  
}
