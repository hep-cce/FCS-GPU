#include "AtlasStyle.C"
void rootlogon()
{
  // Load ATLAS style
  //gROOT->LoadMacro("AtlasStyle.C"); //No longer works for ROOT6
  SetAtlasStyle();
}
