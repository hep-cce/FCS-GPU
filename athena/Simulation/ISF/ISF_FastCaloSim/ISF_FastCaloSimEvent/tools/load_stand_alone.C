/*
  Copyright (C) 2002-2017 CERN for the benefit of the ATLAS collaboration
*/

{
  gInterpreter->AddIncludePath("..");
  TLorentzVector tdummy;
  TMatrixF fdummy(1,1);
  gROOT->ProcessLine(".L ../src/TFCS1DFunction.cxx+");//TFCS1DFunction.cxx++dvc
}
