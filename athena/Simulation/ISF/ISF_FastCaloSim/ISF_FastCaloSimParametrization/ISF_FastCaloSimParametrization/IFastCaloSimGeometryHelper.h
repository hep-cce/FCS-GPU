/*
  Copyright (C) 2002-2017 CERN for the benefit of the ATLAS collaboration
*/

///////////////////////////////////////////////////////////////////
// IFastCaloSimGeometryHelper.h, (c) ATLAS Detector software
///////////////////////////////////////////////////////////////////

#ifndef IFastCaloSimGeometryHelper_H
#define IFastCaloSimGeometryHelper_H 1

// Gaudi
#include "GaudiKernel/IAlgTool.h"
#include "ISF_FastCaloSimEvent/ICaloGeometry.h"
 
static const InterfaceID IID_IFastCaloSimGeometryHelper("IFastCaloSimGeometryHelper", 1, 0);
   
class IFastCaloSimGeometryHelper : virtual public IAlgTool,virtual public ICaloGeometry {
  public:

  virtual ~IFastCaloSimGeometryHelper() {}
  /** AlgTool interface methods */
  static const InterfaceID& interfaceID() { return IID_IFastCaloSimGeometryHelper; }
    
};

#endif // IFastCaloSimGeometryHelper_H

