/*
  Copyright (C) 2002-2023 CERN for the benefit of the ATLAS collaboration
*/

#ifndef GEOGPU_STRUCTS_H
#define GEOGPU_STRUCTS_H

#include "GeoRegion.h"

struct Rg_Sample_Index {
        int size ;
        int index ;
};


struct GeoGpu {
	unsigned long ncells ;
	CaloDetDescrElement* cells; 
	unsigned int nregions ;
	GeoRegion* regions ;
	int max_sample ;
	Rg_Sample_Index * sample_index ;
};	

#endif
