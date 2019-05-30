/*
  Copyright (C) 2002-2017 CERN for the benefit of the ATLAS collaboration
*/

#ifndef FastCaloSimGeometryHelper_h
#define FastCaloSimGeometryHelper_h

// Athena includes
#include "AthenaBaseComps/AthAlgTool.h"

#include "ISF_FastCaloSimParametrization/CaloGeometry.h"
#include "ISF_FastCaloSimParametrization/IFastCaloSimGeometryHelper.h"

class CaloDetDescrManager;
class IGeoModelSvc;

class FastCaloSimGeometryHelper:public AthAlgTool, public CaloGeometry, virtual public IFastCaloSimGeometryHelper {
  public :
    /** Constructor with parameters */
    FastCaloSimGeometryHelper( const std::string& t, const std::string& n, const IInterface* p );

    /** Destructor */
  virtual ~FastCaloSimGeometryHelper();

    // Athena algtool's Hooks
  virtual StatusCode  initialize() override final;
  virtual StatusCode  finalize() override final;

  virtual StatusCode geoInit(IOVSVC_CALLBACK_ARGS);
  
  private:  
    const IGeoModelSvc *m_geoModel;
    /// DetDescr mgr for access to the calo helper
    const CaloDetDescrManager* m_caloMgr;  
    
    virtual bool LoadGeometryFromCaloDDM();
};

#endif

