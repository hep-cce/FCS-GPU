/*
  Copyright (C) 2002-2018 CERN for the benefit of the ATLAS collaboration
*/

#ifndef FCS_SAMPLECONSTANTS_H
#define FCS_SAMPLECONSTANTS_H

#include <string>


namespace
{
#ifdef FCS_INPUT_PATH

  static const std::string path_simul( FCS_INPUT_PATH );
  static const std::string path_ahasib( FCS_INPUT_PATH );
  static const std::string path_ahasib_legacy( FCS_INPUT_PATH );

#else

  static const std::string path_simul( "/eos/atlas/atlascerngroupdisk/proj-simul" );
  static const std::string path_ahasib( "/eos/user/a/ahasib/Data" );
  static const std::string path_ahasib_legacy( "/eos/atlas/user/a/ahasib/public/Simul-FastCalo" );

#endif
} // anonymous namespace


namespace FCS
{

  static const std::string VERSION_PARAMETRIZATION = "v010";
  static const std::string VERSION_DSID            = "ver07";
  static const std::string VERSION_FIRSTPCA        = "ver07";
  static const std::string VERSION_WIGGLE          = "ver06";
  static const std::string VERSION_INTERPOLATION   = "ver06";

static const std::string BASEDIR_PARAMETRIZATION
    = path_simul + "/BigParamFiles/";
static const std::string BASEDIR_GEOMETRY
    = path_simul + "/CaloGeometry/";
static const std::string BASEDIR_DSID
    = path_ahasib + "/ParametrizationProduction07/";
static const std::string BASEDIR_INPUTS
    = path_simul + "/InputSamplesProdsysProduction/";
static const std::string BASEDIR_FIRSTPCA
    = path_ahasib + "/ParametrizationProduction07/";
static const std::string BASEDIR_ETASLICE
    = path_ahasib + "/ParametrizationProduction07/";
static const std::string BASEDIR_WIGGLE
    = path_ahasib + "/ParametrizationProduction07/";
static const std::string BASEDIR_INTERPOLATION
    = path_ahasib_legacy + "/ParametrizationProductionVer06/";

} // namespace FCS

#endif // FCS_SAMPLECONSTANTS_H
