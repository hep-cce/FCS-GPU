/*
  Copyright (C) 2002-2021 CERN for the benefit of the ATLAS collaboration
*/

#include "AtlasStyle.C"
void rootlogon()
{
  // Load ATLAS style
  //gROOT->LoadMacro("AtlasStyle.C"); //No longer works for ROOT6
  SetAtlasStyle();
}
