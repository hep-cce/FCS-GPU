/*
  Copyright (C) 2002-2019 CERN for the benefit of the ATLAS collaboration
*/

#ifndef FCS_SAMPLEINFO_H
#define FCS_SAMPLEINFO_H

#include <iostream>
#include <string>

namespace FCS {

struct DSIDInfo {
  DSIDInfo(int dsid_ = -1, int pdgId_ = -1, int energy_ = -1, float eta_ = -1, int zVertex_ = -1)
      : dsid(dsid_), pdgId(pdgId_), energy(energy_), eta(eta_),
        zVertex(zVertex_) {}

  int dsid = -1;
  int pdgId = -1;
  int energy = -1;
  float eta = -1;
  int zVertex = -1;
};

struct SampleInfo {
  SampleInfo(int dsid_ = - 1, std::string location_ = "", std::string label_ = "",
             int pdgId_ = -1, int energy_ = -1,
             float etaMin_ = -1, float etaMax_ = -1, int zVertex_ = -1)
      : dsid(dsid_), location(location_), label(label_),
        pdgId(pdgId_), energy(energy_),
        etaMin(etaMin_), etaMax(etaMax_), zVertex(zVertex_) {}

  int dsid;
  std::string location;
  std::string label;
  int pdgId;
  int energy;
  float etaMin;
  float etaMax;
  int zVertex;
};

std::ostream &operator<<(std::ostream &out,
                         const DSIDInfo &info);

std::ostream &operator<<(std::ostream &out,
                         const SampleInfo &info);

} // namespace FCS

#endif // FCS_SAMPLEINFO_H
