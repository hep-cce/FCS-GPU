/*
  Copyright (C) 2002-2017 CERN for the benefit of the ATLAS collaboration
*/

#include "AtlasStyle.C"
#include "AtlasUtils.C"

void rootlogon()
{
  // Load ATLAS style
  SetAtlasStyle();
  gROOT->SetStyle("ATLAS");
  gROOT->ForceStyle();
 
}
