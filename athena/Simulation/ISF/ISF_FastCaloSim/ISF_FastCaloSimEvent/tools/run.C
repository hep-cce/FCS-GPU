/*
  Copyright (C) 2002-2017 CERN for the benefit of the ATLAS collaboration
*/

{
  TFCS1DFunction f;
  f.dummytest=1234.5;
  TFile::Open("test.root","recreate");
  f.Write("test");
}
