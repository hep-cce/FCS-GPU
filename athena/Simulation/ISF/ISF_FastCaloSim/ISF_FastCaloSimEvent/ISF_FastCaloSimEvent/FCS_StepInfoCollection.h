/*
  Copyright (C) 2002-2017 CERN for the benefit of the ATLAS collaboration
*/

#ifndef ISF_FASTCALOSIMEVENT_ISF_FCS_STEPINFOCOLLECTION_H
#define ISF_FASTCALOSIMEVENT_ISF_FCS_STEPINFOCOLLECTION_H

// athena includes
#include "AthContainers/DataVector.h"

// local include
#include "ISF_FastCaloSimEvent/FCS_StepInfo.h"

// Namespace for the ShowerLib related classes
namespace ISF_FCS_Parametrization {

  /**
   *
   *  @short Class for collection of StepInfo class (G4 hits)
   *          copied and modified version to ISF
   *
   *  @author Wolfgang Ehrenfeld, University of Hamburg, Germany
   *  @author Sasha Glazov, DESY Hamburg, Germany
   *  @author Zdenek Hubacek, CERN
   *
   *  @version $Id: FCS_StepInfoCollection $
   *
   */

  class FCS_StepInfoCollection : public DataVector<FCS_StepInfo> {

  public:

          FCS_StepInfoCollection() {}
  private:

  };

} // namespace ShowerLib


#include "SGTools/CLASS_DEF.h"

CLASS_DEF( ISF_FCS_Parametrization::FCS_StepInfoCollection , 1330006248 , 1 )

SG_BASE(  ISF_FCS_Parametrization::FCS_StepInfoCollection, DataVector<ISF_FCS_Parametrization::FCS_StepInfo>);

#endif
