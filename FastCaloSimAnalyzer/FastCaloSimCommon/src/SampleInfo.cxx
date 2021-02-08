/*
  Copyright (C) 2002-2019 CERN for the benefit of the ATLAS collaboration
*/

#include "SampleInfo.h"

namespace FCS {

  std::ostream& operator<<( std::ostream& out, const DSIDInfo& info ) {
    out << "DSID: " << info.dsid << "\t";
    out << "PDG ID: " << info.pdgId << "\t";
    out << "Energy: " << info.energy << "\t";
    out << "Eta: " << info.eta << "\t";
    out << "Z: " << info.zVertex << "\t";

    return out;
  }

  std::ostream& operator<<( std::ostream& out, const SampleInfo& info ) {
    out << "DSID: " << info.dsid << "\t";
    out << "Location: " << info.location << "\t";

    return out;
  }

} // namespace FCS
