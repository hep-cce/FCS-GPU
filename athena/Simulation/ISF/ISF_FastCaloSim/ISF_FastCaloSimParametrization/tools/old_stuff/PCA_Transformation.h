/*
  Copyright (C) 2002-2017 CERN for the benefit of the ATLAS collaboration
*/

#ifndef PCA_TRANSFORMATION_H
#define PCA_TRANSFORMATION_H

//////////////////////////////////////////////////
//
//		Class PCA_Transformation
//		PCA_Transformation.h
//
//////////////////////////////////////////////////

#include "TMatrixD.h"
#include "TVectorD.h"

class PCA_Transformation{

public:
//Constructeurs
PCA_Transformation();
PCA_Transformation(int NVariables, TMatrixD EigenVectors, TVectorD EigenValues, TVectorD MeanValues, TVectorD SigmaValues);

    
//Services
int get_NVariables();
TMatrixD get_EigenVectors();
TVectorD get_EigenValues();
TVectorD get_MeanValues();
TVectorD get_SigmaValues();
    
void X2P(PCA_Transformation PCA, double *x, double *p);
void P2X(PCA_Transformation PCA, double *p, double *x, int nTest);

private:
    
int NVariables_;
TMatrixD EigenVectors_;
TVectorD EigenValues_;
TVectorD MeanValues_;
TVectorD SigmaValues_;

};

#endif

