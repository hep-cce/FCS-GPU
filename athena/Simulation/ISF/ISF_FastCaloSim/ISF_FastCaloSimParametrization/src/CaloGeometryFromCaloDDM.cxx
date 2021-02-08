/*
  Copyright (C) 2002-2017 CERN for the benefit of the ATLAS collaboration
*/

#include "ISF_FastCaloSimParametrization/CaloGeometryFromCaloDDM.h"
#include "CaloDetDescr/CaloDetDescrElement.h"
//#include "ISF_FastCaloSimParametrization/CaloGeoDetDescrElement.h"
#include "CaloDetDescr/CaloDetDescrManager.h"

using namespace std;

CaloGeometryFromCaloDDM::CaloGeometryFromCaloDDM() : CaloGeometry()
{
}

CaloGeometryFromCaloDDM::~CaloGeometryFromCaloDDM()
{
}

bool CaloGeometryFromCaloDDM::LoadGeometryFromCaloDDM(const CaloDetDescrManager* calo_dd_man)
{
  int jentry=0;
  for(CaloDetDescrManager::calo_element_const_iterator calo_iter=calo_dd_man->element_begin();calo_iter<calo_dd_man->element_end();++calo_iter) {
    const CaloDetDescrElement* pcell=*calo_iter;
    addcell(pcell);
  
    if(jentry%25000==0) {
      cout<<jentry<<" : "<<pcell->getSampling()<<", "<<pcell->identify()<<endl;
    }
    ++jentry;
  }

  return PostProcessGeometry();
}
bool CaloGeometryFromCaloDDM::LoadFCalChannelMapFromFCalDDM(const FCALDetectorManager* fcal_dd_man){
   this->SetFCal_ChannelMap( fcal_dd_man->getChannelMap() );
   this->calculateFCalRminRmax();
   return this->checkFCalGeometryConsistency();
}
