/*
  Copyright (C) 2002-2018 CERN for the benefit of the ATLAS collaboration
*/

#include "ISF_FastCaloSimEvent/TFCSParametrizationBinnedChain.h"
#include "ISF_FastCaloSimEvent/TFCSSimulationState.h"
#include "ISF_FastCaloSimEvent/TFCSTruthState.h"
#include "ISF_FastCaloSimEvent/TFCSExtrapolationState.h"
#include <algorithm>
#include <iterator>
#include <iostream>

//=============================================
//======= TFCSParametrizationBinnedChain =========
//=============================================

void TFCSParametrizationBinnedChain::push_before_first_bin(TFCSParametrizationBase* param) 
{
  if(m_bin_start[0]==size()) {
    chain().push_back(param);
  } else {  
    Chain_t::iterator it(&chain()[m_bin_start[0]]);
    chain().insert(it,param);
  }  
  for(auto& ele:m_bin_start) ele++;
  recalc();
}

void TFCSParametrizationBinnedChain::push_back_in_bin(TFCSParametrizationBase* param, unsigned int bin) 
{
  if(m_bin_start.size()<=bin+1) {
    m_bin_start.resize(bin+2,m_bin_start.back());
    m_bin_start.shrink_to_fit();
  }  
  if(m_bin_start[bin+1]==size()) {
    chain().push_back(param);
  } else {  
    Chain_t::iterator it(&chain()[m_bin_start[bin+1]]);
    chain().insert(it,param);
  }  
  for(unsigned int ibin=bin+1;ibin<m_bin_start.size();++ibin) m_bin_start[ibin]++;
  recalc();
}

int TFCSParametrizationBinnedChain::get_bin(TFCSSimulationState&,const TFCSTruthState*, const TFCSExtrapolationState*) const
{
  return 0;
}

const std::string TFCSParametrizationBinnedChain::get_variable_text(TFCSSimulationState&,const TFCSTruthState*, const TFCSExtrapolationState*) const
{
  return std::string("NO VARIABLE DEFINED");
}

const std::string TFCSParametrizationBinnedChain::get_bin_text(int bin) const
{
  return std::string(Form("bin %d",bin));
}

FCSReturnCode TFCSParametrizationBinnedChain::simulate(TFCSSimulationState& simulstate,const TFCSTruthState* truth, const TFCSExtrapolationState* extrapol)
{
  for(unsigned int ichain=0;ichain<m_bin_start[0];++ichain) {
    ATH_MSG_DEBUG("now run for all bins: "<<chain()[ichain]->GetName());
    if (simulate_and_retry(chain()[ichain], simulstate, truth, extrapol) != FCSSuccess) {
      return FCSFatal;
    }
  }
  if(get_number_of_bins()>0) {
    int bin=get_bin(simulstate,truth,extrapol);
    if(bin>=0 && bin<(int)get_number_of_bins()) {
      for(unsigned int ichain=m_bin_start[bin];ichain<m_bin_start[bin+1];++ichain) {
        ATH_MSG_DEBUG("for "<<get_variable_text(simulstate,truth,extrapol)<<" run "<<get_bin_text(bin)<<": "<<chain()[ichain]->GetName());
        if (simulate_and_retry(chain()[ichain], simulstate, truth, extrapol) != FCSSuccess) {
          return FCSFatal;
        }
      }
    } else {
      ATH_MSG_WARNING("for "<<get_variable_text(simulstate,truth,extrapol)<<": "<<get_bin_text(bin));
    }
  } else {
    ATH_MSG_WARNING("no bins defined, is this intended?");
  }  
  for(unsigned int ichain=m_bin_start.back();ichain<size();++ichain) {
    ATH_MSG_DEBUG("now run for all bins: "<<chain()[ichain]->GetName());
    if (simulate_and_retry(chain()[ichain], simulstate, truth, extrapol) != FCSSuccess) {
      return FCSFatal;
    }
  }

  return FCSSuccess;
}

void TFCSParametrizationBinnedChain::Print(Option_t *option) const
{
  TFCSParametrization::Print(option);
  TString opt(option);
  bool shortprint=opt.Index("short")>=0;
  bool longprint=msgLvl(MSG::DEBUG) || (msgLvl(MSG::INFO) && !shortprint);
  TString optprint=opt;optprint.ReplaceAll("short","");

  TString prefix="- ";
  for(unsigned int ichain=0;ichain<size();++ichain) {
    if(ichain==0 && ichain!=m_bin_start.front()) {
      prefix="> ";
      if(longprint) ATH_MSG_INFO(optprint<<prefix<<"Run for all bins");
    }  
    for(unsigned int ibin=0;ibin<get_number_of_bins();++ibin) {
      if(ichain==m_bin_start[ibin]) {
        if(ibin<get_number_of_bins()-1) if(ichain==m_bin_start[ibin+1]) continue;
        prefix=Form("%-2d",ibin);
        if(longprint) ATH_MSG_INFO(optprint<<prefix<<"Run for "<<get_bin_text(ibin));
      }  
    }  
    if(ichain==m_bin_start.back()) {
      prefix="< ";
      if(longprint) ATH_MSG_INFO(optprint<<prefix<<"Run for all bins");
    }  
    chain()[ichain]->Print(opt+prefix);
  }
}

void TFCSParametrizationBinnedChain::unit_test(TFCSSimulationState* simulstate,const TFCSTruthState* truth, const TFCSExtrapolationState* extrapol)
{
  if(!simulstate) simulstate=new TFCSSimulationState();
  if(!truth) truth=new TFCSTruthState();
  if(!extrapol) extrapol=new TFCSExtrapolationState();

  TFCSParametrizationBinnedChain chain("chain","chain");
  chain.setLevel(MSG::DEBUG);

  std::cout<<"====         Chain setup       ===="<<std::endl;
  chain.Print();
  std::cout<<"==== Simulate with empty chain ===="<<std::endl;
  chain.simulate(*simulstate,truth,extrapol);
  std::cout<<"==================================="<<std::endl<<std::endl;

  TFCSParametrizationBase* param;
  param=new TFCSParametrization("A begin all","A begin all");
  param->setLevel(MSG::DEBUG);
  chain.push_before_first_bin(param);
  param=new TFCSParametrization("A end all","A end all");
  param->setLevel(MSG::DEBUG);
  chain.push_back(param);

  std::cout<<"====         Chain setup       ===="<<std::endl;
  chain.Print();
  std::cout<<"==== Simulate only begin/end all ===="<<std::endl;
  chain.simulate(*simulstate,truth,extrapol);
  std::cout<<"==================================="<<std::endl<<std::endl;

  for(int i=0;i<3;++i) {
    TFCSParametrizationBase* param=new TFCSParametrization(Form("A%d",i),Form("A %d",i));
    param->setLevel(MSG::DEBUG);
    chain.push_back_in_bin(param,i);
  }

  for(int i=3;i>0;--i) {
    TFCSParametrizationBase* param=new TFCSParametrization(Form("B%d",i),Form("B %d",i));
    param->setLevel(MSG::DEBUG);
    chain.push_back_in_bin(param,i);
  }
  param=new TFCSParametrization("B end all","B end all");
  param->setLevel(MSG::DEBUG);
  chain.push_back(param);
  param=new TFCSParametrization("B begin all","B begin all");
  param->setLevel(MSG::DEBUG);
  chain.push_before_first_bin(param);
  
  std::cout<<"====         Chain setup       ===="<<std::endl;
  chain.Print();
  std::cout<<"==== Simulate with full chain  ===="<<std::endl;
  chain.simulate(*simulstate,truth,extrapol);
  std::cout<<"==================================="<<std::endl<<std::endl;
}
