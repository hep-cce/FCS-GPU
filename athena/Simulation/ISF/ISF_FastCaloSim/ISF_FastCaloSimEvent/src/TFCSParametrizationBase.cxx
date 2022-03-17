/*
  Copyright (C) 2002-2018 CERN for the benefit of the ATLAS collaboration
*/

#include "ISF_FastCaloSimEvent/TFCSParametrizationBase.h"
#include "TClass.h"

//=============================================
//======= TFCSParametrizationBase =========
//=============================================

std::set<int> TFCSParametrizationBase::s_no_pdgid;
std::vector<TFCSParametrizationBase*> TFCSParametrizationBase::s_cleanup_list;

#ifndef __FastCaloSimStandAlone__
// Initialize only in constructor to make sure the needed services are ready
Athena::MsgStreamMember* TFCSParametrizationBase::s_msg(nullptr);
#endif

#if defined(__FastCaloSimStandAlone__)
TFCSParametrizationBase::TFCSParametrizationBase(const char* name,
                                                 const char* title)
    : TNamed(name, title), m_level(MSG::INFO), m_msg(&std::cout) {}
#else
TFCSParametrizationBase::TFCSParametrizationBase(const char* name,
                                                 const char* title)
    : TNamed(name, title) {
  if (s_msg == nullptr)
    s_msg = new Athena::MsgStreamMember("FastCaloSimParametrization");
}
#endif

void TFCSParametrizationBase::set_geometry(ICaloGeometry* geo) {
  for (unsigned int i = 0; i < size(); ++i) (*this)[i]->set_geometry(geo);
}

/// Result should be returned in simulstate.
/// Simulate all energies in calo layers for energy parametrizations.
/// Simulate cells for shape simulation.
FCSReturnCode TFCSParametrizationBase::simulate(
    TFCSSimulationState& /*simulstate*/, const TFCSTruthState* /*truth*/,
    const TFCSExtrapolationState* /*extrapol*/) {
  ATH_MSG_ERROR(
      "now in TFCSParametrizationBase::simulate(). This should normally not "
      "happen");
  return FCSFatal;
}

/// If called with argument "short", only a one line summary will be printed
void TFCSParametrizationBase::Print(Option_t* option) const {
  TString opt(option);
  bool shortprint = opt.Index("short") >= 0;
  bool longprint = msgLvl(MSG::DEBUG) || (msgLvl(MSG::INFO) && !shortprint);
  TString optprint = opt;
  optprint.ReplaceAll("short", "");

  if (longprint) {
    ATH_MSG_INFO(optprint << GetTitle() << " (" << IsA()->GetName() << "*)"
                          << this);
    ATH_MSG(INFO) << optprint << "  PDGID: ";
    if (is_match_all_pdgid()) {
      msg() << "all";
    } else {
      for (std::set<int>::iterator it = pdgid().begin(); it != pdgid().end();
           ++it) {
        if (it != pdgid().begin()) msg() << ", ";
        msg() << *it;
      }
    }
    if (is_match_all_Ekin()) {
      msg() << " ; Ekin=all";
    } else {
      msg() << " ; Ekin=" << Ekin_nominal() << " [" << Ekin_min() << " , "
            << Ekin_max() << ") MeV";
    }
    if (is_match_all_eta()) {
      msg() << " ; eta=all";
    } else {
      msg() << " ; eta=" << eta_nominal() << " [" << eta_min() << " , "
            << eta_max() << ")";
    }
    msg() << endmsg;
  } else {
    ATH_MSG_INFO(optprint << GetTitle());
  }
}

void TFCSParametrizationBase::DoCleanup() {
  // Do cleanup only at the end of read/write operations
  for (auto ptr : s_cleanup_list)
    if (ptr) {
      delete ptr;
    }
  s_cleanup_list.resize(0);
}
