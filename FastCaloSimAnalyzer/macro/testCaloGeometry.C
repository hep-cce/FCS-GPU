/*
  Copyright (C) 2002-2017 CERN for the benefit of the ATLAS collaboration
*/
#include "CaloGeometryFromFile.h"


void testCaloGeometry();

void testCaloGeometry()
{

  CaloGeometryFromFile* geo = new CaloGeometryFromFile();

// * load geometry files
  geo->LoadGeometryFromFile("/afs/cern.ch/atlas/groups/Simulation/FastCaloSimV2/Geometry-ATLAS-R2-2016-01-00-01.root", "ATLAS-R2-2016-01-00-01");
  TString path_to_fcal_geo_files = "/afs/cern.ch/atlas/groups/Simulation/FastCaloSimV2/";
  geo->LoadFCalGeometryFromFiles(path_to_fcal_geo_files + "FCal1-electrodes.sorted.HV.09Nov2007.dat", path_to_fcal_geo_files + "FCal2-electrodes.sorted.HV.April2011.dat", path_to_fcal_geo_files + "FCal3-electrodes.sorted.HV.09Nov2007.dat");


  const CaloDetDescrElement* cell;
  cell = geo->getDDE(2, 0.24, 0.24); //(layer, eta, phi)


  cout << "Found cell id=" << cell->identify() << " sample=" << cell->getSampling() << " eta=" << cell->eta() << " phi=" << cell->phi() << endl;

  cell = geo->getDDE(3260641881524011008);

  cout << "Found cell id=" << cell->identify() << " sample=" << cell->getSampling() << " eta=" << cell->eta() << " phi=" << cell->phi() << endl;



}

