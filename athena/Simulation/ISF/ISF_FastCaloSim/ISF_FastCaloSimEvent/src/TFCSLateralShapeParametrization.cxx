/*
  Copyright (C) 2002-2018 CERN for the benefit of the ATLAS collaboration
*/

#include "ISF_FastCaloSimEvent/TFCSLateralShapeParametrization.h"
#include "ISF_FastCaloSimEvent/FastCaloSim_CaloCell_ID.h"

//=============================================
//======= TFCSLateralShapeParametrization =========
//=============================================

TFCSLateralShapeParametrization::TFCSLateralShapeParametrization(const char* name, const char* title):TFCSParametrization(name,title),m_Ekin_bin(-1),m_calosample(-1)
{
}

void TFCSLateralShapeParametrization::set_Ekin_bin(int bin)
{
  m_Ekin_bin=bin;
}

void TFCSLateralShapeParametrization::set_calosample(int cs)
{
  m_calosample=cs;
}

void TFCSLateralShapeParametrization::set_pdgid_Ekin_eta_Ekin_bin_calosample(const TFCSLateralShapeParametrization& ref)
{
  set_calosample(ref.calosample());
  set_Ekin_bin(ref.Ekin_bin());
  set_pdgid_Ekin_eta(ref);
}

void TFCSLateralShapeParametrization::Print(Option_t *option) const
{
  TString opt(option);
  bool shortprint=opt.Index("short")>=0;
  bool longprint=msgLvl(MSG::DEBUG) || (msgLvl(MSG::INFO) && !shortprint);
  TString optprint=opt;optprint.ReplaceAll("short","");
  TFCSParametrization::Print(option);
  if(longprint) {
    if(Ekin_bin()==-1 ) ATH_MSG_INFO(optprint <<"  Ekin_bin=all ; calosample="<<calosample());
     else               ATH_MSG_INFO(optprint <<"  Ekin_bin="<<Ekin_bin()<<" ; calosample="<<calosample());
  }  
}
