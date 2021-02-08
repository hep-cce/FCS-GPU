/*
  Copyright (C) 2002-2017 CERN for the benefit of the ATLAS collaboration
*/

using namespace std;
#include "ISF_FastCaloSimEvent/TFCS1DFunctionRegressionTF.h"
#include "TFile.h"
#include "TString.h"
#include "TMath.h"


//=============================================
//======= TFCS1DFunctionRegressionTF =========
//=============================================

using namespace std;

TFCS1DFunctionRegressionTF::TFCS1DFunctionRegressionTF(float rangeval, float startval)
{
  m_rangeval = rangeval;
  m_startval = startval;
}


double TFCS1DFunctionRegressionTF::retransform(double value) const
{
 
 return (value*m_rangeval+m_startval);
 
}

double TFCS1DFunctionRegressionTF::rnd_to_fct(double rnd) const
{
  
  double value=regression_value(rnd);
  if(m_rangeval>0)
   value=retransform(value);
  return value;
  
}

