/*
  Copyright (C) 2002-2018 CERN for the benefit of the ATLAS collaboration
*/

#ifndef FCS_SAMPLECONSTANTS_H
#define FCS_SAMPLECONSTANTS_H

#include <string>

namespace FCS
{

  static const std::string VERSION_PARAMETRIZATION = "v010";
  static const std::string VERSION_DSID            = "ver07";
  static const std::string VERSION_FIRSTPCA        = "ver07";
  static const std::string VERSION_WIGGLE          = "ver06";
  static const std::string VERSION_INTERPOLATION   = "ver06";

  static const std::string DIR_PARAMETRIZATION = "BigParamFiles/";
  static const std::string DIR_GEOMETRY        = "CaloGeometry/";
  static const std::string DIR_DSID            = "ParametrizationProduction07/";
  static const std::string DIR_INPUTS          = "InputSamplesProdsysProduction/";
  static const std::string DIR_FIRSTPCA        = "ParametrizationProduction07/";
  static const std::string DIR_ETASLICE        = "ParametrizationProduction07/";
  static const std::string DIR_WIGGLE          = "ParametrizationProduction07/";
  static const std::string DIR_INTERPOLATION   = "ParametrizationProductionVer06/";
  
} // namespace FCS

#endif // FCS_SAMPLECONSTANTS_H
