/*
  Copyright (C) 2002-2018 CERN for the benefit of the ATLAS collaboration
*/

#ifndef TFCS_SAMPLEDISCOVERY_H
#define TFCS_SAMPLEDISCOVERY_H

#include <array>
#include <vector>

#include "SampleConstants.h"
#include "SampleInfo.h"

class TFCSSampleDiscovery {
public:
  TFCSSampleDiscovery( std::string fileName = "DSID_DB.txt", bool debug = false );
  ~TFCSSampleDiscovery() = default;

  const FCS::DSIDInfo& findDSID( int pdgId, int energy, float eta, int zVertex ) const;

  int getEnergy( int dsid ) const;
  int getPdgId( int dsid ) const;

  FCS::SampleInfo findSample( int inDSID, const std::string& fileName = "InputSamplesList.txt" ) const;

  std::string        getBaseName( int dsid ) const;
  std::string        getFirstPCAAppName( int dsid, const std::string& basedir = FCS::BASEDIR_FIRSTPCA,
                                         const std::string& version = FCS::VERSION_FIRSTPCA ) const;
  std::string        getSecondPCAName( int dsid, const std::string& basedir = FCS::BASEDIR_DSID,
                                       const std::string& version = FCS::VERSION_DSID ) const;
  std::string        getShapeName( int dsid, const std::string& basedir = FCS::BASEDIR_DSID,
                                   const std::string& version = FCS::VERSION_DSID ) const;
  std::string        getAvgSimShapeName( int dsid, const std::string& basedir = FCS::BASEDIR_DSID,
                                         const std::string& version = FCS::VERSION_DSID ) const;
  std::string        getEinterpolMeanName( int pdgId, const std::string& basedir = FCS::BASEDIR_INTERPOLATION,
                                           const std::string& version = FCS::VERSION_INTERPOLATION ) const;
  static std::string getParametrizationName( const std::string& basedir = FCS::BASEDIR_PARAMETRIZATION,
                                             const std::string& version = FCS::VERSION_PARAMETRIZATION );
  static std::string getWiggleName( const std::string& etaRange, int sampling, bool isNewWiggle = true,
                                    const std::string& basedir = FCS::BASEDIR_WIGGLE,
                                    const std::string& version = FCS::VERSION_WIGGLE );

  // Geometry
  static std::string                geometryTree();
  static std::string                geometryName( const std::string& basedir = FCS::BASEDIR_GEOMETRY );
  static std::array<std::string, 3> geometryNameFCal( const std::string& basedir = FCS::BASEDIR_GEOMETRY );
  static std::string                geometryMap( const std::string& basedir = FCS::BASEDIR_GEOMETRY );

private:
  std::ifstream      openFile( const std::string& fileName ) const;
  static std::string prefixLocation( const std::string& location );

  std::string getName( int dsid, const std::string& label, const std::string& basedir,
                       const std::string& version ) const;

  const FCS::DSIDInfo        m_invalid;
  std::vector<FCS::DSIDInfo> m_dbDSID;
};

#endif // TFCS_SAMPLEDISCOVERY_H
