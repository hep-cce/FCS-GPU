/*
  Copyright (C) 2002-2017 CERN for the benefit of the ATLAS collaboration
*/

#ifndef ISF_FASTCALOSIMEVENT_ISF_FCS_STEPINFO_H
#define ISF_FASTCALOSIMEVENT_ISF_FCS_STEPINFO_H

// STL includes
#include <iostream>
#include <vector>
#include <string>

// CLHEP include for Hep3Vector
#include "CLHEP/Vector/ThreeVector.h"

//include for Hit
//#include "LArSimEvent/LArHit.h"
#include "TileSimEvent/TileHit.h"

class MsgStream;

// Namespace for the G4 step related classes
namespace ISF_FCS_Parametrization {

  /**
   *
   *   @short Class to collect information about G4 steps
   *
   *          This class is designed to transfer hit information,
   *          i.e. position, energy deposition and time, from
   *          G4 simulation to the FastCaloSim parametrization
   *
   *  @author Wolfgang Ehrenfeld, University of Hamburg, Germany
   *  @author Sasha Glazov, DESY Hamburg, Germany
   *  @author Zdenek Hubacek, CERN
   *
   * @version $Id: FCS_StepInfo $
   *
   */

  class FCS_StepInfo: public TileHit {

  public:

    //! empty default constructor
    FCS_StepInfo(): m_pos(), m_valid(false), m_detector(-1) {}

    FCS_StepInfo(CLHEP::Hep3Vector l_vec, Identifier l_cell, double l_energy, double l_time, bool l_valid, int l_detector): TileHit(l_cell,l_energy, l_time) { m_pos = l_vec; m_valid = l_valid, m_detector = l_detector; }

    //FCS_StepInfo(const FCS_StepInfo& first, const FCS_StepInfo& second);

    /* access functions */

    //! set position
    inline void setP(const CLHEP::Hep3Vector& p) { m_pos = p; }
    //! set x position
    inline void setX(const double x) { return m_pos.setX(x); }
    //! set y position
    inline void setY(const double y) { return m_pos.setY(y); }
    //! set z position
    inline void setZ(const double z) { return m_pos.setZ(z); }
    //! set depoisted energy
    //inline void setE(const double t) { m_energy = t; }
    //! set time
    //inline void setTime(const double t) { m_time = t; }
    //! set validity
    inline void setValid(const bool flag) { m_valid = flag; }
    //! set identifier
    //inline void setIdentifier(const Identifier id) { m_ID = id; }

    inline void setDetector(const int det) { m_detector = det; }
    //! return spacial position
    inline CLHEP::Hep3Vector position() const { return m_pos; }
    //! return x position
    inline double x() const { return m_pos.x(); }
    //! return y position
    inline double y() const { return m_pos.y(); }
    //! return z position
    inline double z() const { return m_pos.z(); }
    //! return deposited energy
    //inline double dep() const { return m_energy; }
    //! return time of hit
    //inline double time() const { return time(); }
    //! return validity flag
    inline bool valid() const { return m_valid; }

    inline int detector() const { return m_detector; }

    /* helper functions */

    //! return spactial distance squared
    double diff2(const FCS_StepInfo& other) const;

    //! energy weighted sum
    FCS_StepInfo& operator+=(const FCS_StepInfo& other );

  private:

    // data members
    CLHEP::Hep3Vector m_pos;    //!< spatial position
    bool m_valid;        //!< flag, if hit is valid (if valid calculator?)
    int m_detector;      //!< dictionary value in which detector the hit is

  };

} // namespace ISF_FCS_Parametrization

#endif
