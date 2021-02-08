/*
  Copyright (C) 2002-2018 CERN for the benefit of the ATLAS collaboration
*/

#include "ISF_FastCaloSimEvent/TFCSParametrizationEbinChain.h"
#include "ISF_FastCaloSimEvent/TFCSLateralShapeParametrization.h"
#include "ISF_FastCaloSimEvent/TFCSSimulationState.h"
#include "ISF_FastCaloSimEvent/TFCSTruthState.h"
#include "ISF_FastCaloSimEvent/TFCSExtrapolationState.h"
#include <iostream>

//=============================================
//======= TFCSParametrizationEbinChain =========
//=============================================

const std::string TFCSParametrizationEbinChain::get_variable_text(TFCSSimulationState& simulstate,const TFCSTruthState*, const TFCSExtrapolationState*) const
{
  return std::string(Form("Ebin=%d",simulstate.Ebin()));
}

void TFCSParametrizationEbinChain::unit_test(TFCSSimulationState* simulstate,const TFCSTruthState* truth, const TFCSExtrapolationState* extrapol)
{
  if(!simulstate) simulstate=new TFCSSimulationState();
  if(!truth) truth=new TFCSTruthState();
  if(!extrapol) extrapol=new TFCSExtrapolationState();

  TFCSParametrizationEbinChain chain("chain","chain");
  chain.setLevel(MSG::DEBUG);

  TFCSParametrizationBase* param;
  param=new TFCSParametrization("A begin all","A begin all");
  param->setLevel(MSG::DEBUG);
  chain.push_before_first_bin(param);
  param=new TFCSParametrization("A end all","A end all");
  param->setLevel(MSG::DEBUG);
  chain.push_back(param);

  for(int i=0;i<3;++i) {
    TFCSLateralShapeParametrization* param=new TFCSLateralShapeParametrization(Form("A%d",i),Form("A %d",i));
    param->setLevel(MSG::DEBUG);
    param->set_Ekin_bin(i);
    chain.push_back_in_bin(param,i);
  }

  for(int i=3;i>0;--i) {
    TFCSLateralShapeParametrization* param=new TFCSLateralShapeParametrization(Form("B%d",i),Form("B %d",i));
    param->setLevel(MSG::DEBUG);
    param->set_Ekin_bin(i);
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
  std::cout<<"==== Simulate with Ebin=0      ===="<<std::endl;
  simulstate->set_Ebin(0);
  chain.simulate(*simulstate,truth,extrapol);
  std::cout<<"==== Simulate with Ebin=1      ===="<<std::endl;
  simulstate->set_Ebin(1);
  chain.simulate(*simulstate,truth,extrapol);
  std::cout<<"==== Simulate with Ebin=2      ===="<<std::endl;
  simulstate->set_Ebin(2);
  chain.simulate(*simulstate,truth,extrapol);
  std::cout<<"==================================="<<std::endl<<std::endl;
}


