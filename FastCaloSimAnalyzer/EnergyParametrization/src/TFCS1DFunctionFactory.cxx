/*
  Copyright (C) 2002-2021 CERN for the benefit of the ATLAS collaboration
*/

#include "ISF_FastCaloSimEvent/TFCS1DFunctionRegression.h"
#include "ISF_FastCaloSimEvent/TFCS1DFunctionRegressionTF.h"
#include "ISF_FastCaloSimEvent/TFCS1DFunctionHistogram.h"
#include "ISF_FastCaloSimEvent/TFCS1DFunction.h"

#include "TFCS1DFunctionFactory.h"
#include "TFCS1DRegression.h"

#include "TFile.h"
#include "TRandom1.h"

#include <iostream>

//#include <string>
#include <sstream>

using namespace std;

//=============================================
//======= TFCS1DFunctionFactory =========
//=============================================


TFCS1DFunction* TFCS1DFunctionFactory::Create(TH1* hist,int skip_regression,int neurons_start, int neurons_end, double maxdev_regression, double maxdev_smartrebin, int ntoys)
{
 
 // This function is called by the user when he wants a histogram to be transformed into a space efficient variant for the parametrization.
 // All code that decides whether a histogram should be transformed into a TFCS1DFunctionRegression or TFCS1DFunctionHistogram
 // should go here. 

  TRandom1* random = new TRandom1();
  random->SetSeed( 0 );
  int          myrand = floor( random->Uniform() * 1000000 );
  stringstream ss;
  ss << myrand;
  string myrandstr = ss.str();

  string xmlweightfilename = "regressionweights" + myrandstr;
  string outfilename       = "TMVAReg" + myrandstr + ".root";
  float  rangeval, startval;

  TFCS1DFunctionRegression*   freg   = 0;
  TFCS1DFunctionRegressionTF* fregTF = 0;
  TFCS1DFunctionHistogram*    fhis   = 0;

  int status = 3;

  if ( !skip_regression )
  status=TFCS1DRegression::testHisto(hist,xmlweightfilename,rangeval,startval,outfilename,neurons_start,neurons_end,maxdev_regression,ntoys);

  cout << "--- testHisto status=" << status << endl;
 if(status==1)
 {
    cout << "Regression" << endl;
    freg = new TFCS1DFunctionRegression();
    vector<vector<double>> fWeightMatrix0to1;
    vector<vector<double>> fWeightMatrix1to2;
    TFCS1DRegression::storeRegression( xmlweightfilename, fWeightMatrix0to1, fWeightMatrix1to2 );
    freg->set_weights( fWeightMatrix0to1, fWeightMatrix1to2 );
    remove( outfilename.c_str() );
    remove( Form( "dl/%s/TMVARegression_MLP.weights.xml", xmlweightfilename.c_str() ) );
    remove( Form( "dl/%s", xmlweightfilename.c_str() ) );
    return freg;
  }
 if(status==2)
 {
    cout << "Regression and Transformation" << endl;
    fregTF = new TFCS1DFunctionRegressionTF( rangeval, startval );
    vector<vector<double>> fWeightMatrix0to1;
    vector<vector<double>> fWeightMatrix1to2;
    TFCS1DRegression::storeRegression( xmlweightfilename, fWeightMatrix0to1, fWeightMatrix1to2 );
    fregTF->set_weights( fWeightMatrix0to1, fWeightMatrix1to2 );
    remove( outfilename.c_str() );
    remove( Form( "dl/%s/TMVARegression_MLP.weights.xml", xmlweightfilename.c_str() ) );
    remove( Form( "dl/%s", xmlweightfilename.c_str() ) );
    return fregTF;
  }
 if(status==3)
 {
    cout << "xmlweightfilename: " << xmlweightfilename << endl;
    remove( outfilename.c_str() );
    remove( Form( "dl/%s/TMVARegression_MLP.weights.xml", xmlweightfilename.c_str() ) );
    remove( Form( "dl/%s", xmlweightfilename.c_str() ) );
    cout << "Histogram" << endl;
    fhis = new TFCS1DFunctionHistogram( hist, maxdev_smartrebin );
    return fhis;
  }
 if(status==0)
 {
    cout << "something went wrong :(" << endl;
    return 0;
  }

  return 0;
}
