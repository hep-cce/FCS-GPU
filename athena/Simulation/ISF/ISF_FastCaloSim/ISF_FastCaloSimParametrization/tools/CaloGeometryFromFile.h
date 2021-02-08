/*
  Copyright (C) 2002-2017 CERN for the benefit of the ATLAS collaboration
*/

#ifndef CaloGeometryFromFile_h
#define CaloGeometryFromFile_h

#include "ISF_FastCaloSimParametrization/CaloGeometry.h"
#include "TString.h"

class CaloGeometryFromFile:public CaloGeometry {
public :
   CaloGeometryFromFile();
   virtual ~CaloGeometryFromFile();
   
   virtual bool LoadGeometryFromFile(TString filename,TString treename,TString hashfile="/afs/cern.ch/atlas/groups/Simulation/FastCaloSimV2/cellId_vs_cellHashId_map.txt");
   virtual bool LoadFCalGeometryFromFiles(TString filename1,TString filename2,TString filename3); // Initialize m_FCal_ChannelMap
   void DrawFCalGraph(int isam,int color);
};

#endif

