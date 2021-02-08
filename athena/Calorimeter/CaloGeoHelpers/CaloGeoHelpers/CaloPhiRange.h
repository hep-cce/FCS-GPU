/*
  Copyright (C) 2002-2017 CERN for the benefit of the ATLAS collaboration
*/

/**
 * @file CaloPhiRange.h
 *
 * @brief CaloPhiRange class declaration
 *
 * $Id: CaloPhiRange.h 587585 2014-03-14 08:32:57Z krasznaa $
 */

#ifndef CALOGEOHELPER_CALOPHIRANGE_H
#define CALOGEOHELPER_CALOPHIRANGE_H

#include <cmath>

/** @class CaloPhiRange
 *
 *  @brief This class defines the phi convention for Calorimeters
 *
 *       up to Release 8.3.0 (included) : 0->2pi
 *
 *       as of 8.4.0 : -pi -> pi
 *
 */


class CaloPhiRange
{
public:
    static double twopi ();
    static double phi_min ();
    static double phi_max ();

    static double fix ( double phi );

    /** @brief simple phi1 - phi2 calculation, but result is fixed to respect range.
     */
    static double diff ( double phi1,  double phi2 );

    static void print();

private:
    
    // This is the real hard-coded choice :
#if 0
    // Doesn't work yet with all the compilers we're supporting.
    static CONSTEXPR double m_phi_min = -M_PI;
    static CONSTEXPR double m_twopi = 2*M_PI;
    static CONSTEXPR double m_phi_max = M_PI;
#endif
    static const double m_phi_min;
    static const double m_twopi;
    static const double m_phi_max;
};

inline double CaloPhiRange::twopi()
{ return m_twopi;}

inline double CaloPhiRange::phi_min()
{ return m_phi_min;}

inline double CaloPhiRange::phi_max()
{ return m_phi_max;}

#endif // CALODETDESCR_CALOPHIRANGE_H
