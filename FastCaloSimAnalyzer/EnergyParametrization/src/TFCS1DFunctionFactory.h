/*
  Copyright (C) 2002-2023 CERN for the benefit of the ATLAS collaboration
*/

#ifndef TFCS1DFunctionFactory_h
#define TFCS1DFunctionFactory_h

#include "ISF_FastCaloSimEvent/TFCS1DFunction.h"

class TFCS1DFunctionFactory
{
  public:
    TFCS1DFunctionFactory() {};
    virtual ~TFCS1DFunctionFactory() {};
    
    static TFCS1DFunction* Create(TH1* hist,int,int,int,double,double,int);
  
  private:

  ClassDef(TFCS1DFunctionFactory,1)
 
};

#if defined(__ROOTCLING__) && defined(__FastCaloSimStandAlone__)
#pragma link C++ class TFCS1DFunctionFactory+;
#endif

#endif
