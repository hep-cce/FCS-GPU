/*
  Copyright (C) 2002-2018 CERN for the benefit of the ATLAS collaboration
*/

#ifndef FCS_dsid_h
#define FCS_dsid_h

#include <vector>
#include <string>

class TChain;

namespace FCS_dsid {
  bool        dsid_is_init           = false;
  std::string FirstPCA_version       = "ver06";
  std::string dsid_version           = "ver06";
  std::string wiggle_version         = "ver06";
  bool        is_NewWiggle           = true;
  std::string dsid_basedir           = "/eos/atlas/user/a/ahasib/public/Simul-FastCalo/ParametrizationProductionVer06/";
  std::string input_basedir          = "/eos/atlas/atlascerngroupdisk/proj-simul/InputSamplesSummer18Complete/";
  std::string FirstPCA_App_basedir   = "/eos/atlas/user/a/ahasib/public/Simul-FastCalo/ParametrizationProductionVer06/";
  std::string param_etaslice_basedir = "/eos/atlas/user/a/ahasib/public/Simul-FastCalo/ParametrizationProductionVer06/";
  std::string param_version          = "ver09";

  void        init();
std::string find_dsid(std::string pdgid, std::string energy, std::string eta,
                      std::string zvertex);
void get_dsid_info(std::string dsid, std::string &pdgid, std::string &energy,
                   std::string &eta, std::string &zvertex);
  std::string get_dsid_basename( std::string dsid );

std::string get_dsid_input_wildcard(std::string dsid,
                                    std::string basedir = input_basedir);
  int         wildcard_add_files_to_chain( TChain* chain, std::string filenames );

std::string get_dsid_shapename(std::string dsid,
                               std::string basedir = dsid_basedir,
                                  std::string version = dsid_version );
std::string get_dsid_avg_sim_shapename(std::string dsid,
                                       std::string basedir = dsid_basedir,
                                          std::string version = dsid_version );
std::string get_dsid_firstPCAname(std::string dsid,
                                  std::string basedir = dsid_basedir,
                                     std::string version = dsid_version );
std::string
get_dsid_firstPCA_Appname(std::string dsid,
                          std::string basedir = FirstPCA_App_basedir,
                                         std::string version = FirstPCA_version );
std::string get_dsid_secondPCAname(std::string dsid,
                                   std::string basedir = dsid_basedir,
                                      std::string version = dsid_version );

std::string get_wiggle_name(std::string etarange, int sampling,
                            bool isNewWiggle = is_NewWiggle,
                            std::string basedir = dsid_basedir,
                            std::string version = wiggle_version);

  std::string get_param_etaslice_name( int pid, std::string etamin, std::string etamax,
                                       std::string basedir = param_etaslice_basedir,
                                       std::string version = param_version );
};

#endif
