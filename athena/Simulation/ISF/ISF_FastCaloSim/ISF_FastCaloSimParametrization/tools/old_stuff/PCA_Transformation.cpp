/*
  Copyright (C) 2002-2017 CERN for the benefit of the ATLAS collaboration
*/

#include "PCA_Transformation.h"
#include "TMatrixD.h"
#include "TVectorD.h"


/////////////////////////////////////////////
PCA_Transformation::PCA_Transformation(){
/////////////////////////////////////////////
    
    TMatrixD initMatrix;
    TVectorD initVector;
    
    NVariables_ = 0;
	EigenVectors_ = initMatrix;
    EigenValues_ = initVector;
    MeanValues_ = initVector;
    SigmaValues_ = initVector;
	
	}

////////////////////////////////////////////////////////////////////////////
PCA_Transformation::PCA_Transformation(int init_NVariables, TMatrixD init_EigenVectors, TVectorD init_EigenValues, TVectorD init_MeanValues, TVectorD init_SigmaValues){
////////////////////////////////////////////////////////////////////////////

    NVariables_ = init_NVariables;
    EigenVectors_ = init_EigenVectors;
    EigenValues_ = init_EigenValues;
    MeanValues_ = init_MeanValues;
    SigmaValues_ = init_SigmaValues;
    
	}


////////////////////////////////////////////////////////
int PCA_Transformation::get_NVariables()
////////////////////////////////////////////////////////
    {return NVariables_;}


////////////////////////////////////////////////////////
TMatrixD PCA_Transformation::get_EigenVectors()
////////////////////////////////////////////////////////
	{return EigenVectors_;}


////////////////////////////////////////////////////////
TVectorD PCA_Transformation::get_EigenValues()
////////////////////////////////////////////////////////
	{return EigenValues_;}


////////////////////////////////////////////////////////
TVectorD PCA_Transformation::get_MeanValues()
////////////////////////////////////////////////////////
    {return MeanValues_;}


////////////////////////////////////////////////////////
TVectorD PCA_Transformation::get_SigmaValues()
////////////////////////////////////////////////////////
    {return SigmaValues_;}


/////////////////////////////////////////////////////////////////////////////
void PCA_Transformation::X2P(PCA_Transformation PCA, double *x, double *p) {
/////////////////////////////////////////////////////////////////////////////
    
    static int gNVariables = PCA.get_NVariables();
    TMatrixD EigenVectors = PCA.get_EigenVectors();
    TVectorD MeanValues = PCA.get_MeanValues();
    TVectorD SigmaValues = PCA.get_SigmaValues();
    
    double* gEigenVectors = EigenVectors.GetMatrixArray();
    double* gMeanValues = MeanValues.GetMatrixArray();
    double* gSigmaValues = SigmaValues.GetMatrixArray();
    
    for (int i = 0; i < gNVariables; i++) {
        p[i] = 0;
        for (int j = 0; j < gNVariables; j++)
            p[i] += (x[j] - gMeanValues[j])
            * gEigenVectors[j *  gNVariables + i] / gSigmaValues[j];
    }
}

/////////////////////////////////////////////////////////////////////////////////////////
void PCA_Transformation::P2X(PCA_Transformation PCA, double *p, double *x, int nTest) {
/////////////////////////////////////////////////////////////////////////////////////////
    
    static int gNVariables = PCA.get_NVariables();
    TMatrixD EigenVectors = PCA.get_EigenVectors();
    TVectorD MeanValues = PCA.get_MeanValues();
    TVectorD SigmaValues = PCA.get_SigmaValues();
    
    double* gEigenVectors = EigenVectors.GetMatrixArray();
    double* gMeanValues = MeanValues.GetMatrixArray();
    double* gSigmaValues = SigmaValues.GetMatrixArray();
    
    for (int i = 0; i < gNVariables; i++) {
        x[i] = gMeanValues[i];
        for (int j = 0; j < nTest; j++)
            x[i] += p[j] * gSigmaValues[i]
            * gEigenVectors[i *  gNVariables + j];
    }
}





