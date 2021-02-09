/*
  Copyright (C) 2002-2021 CERN for the benefit of the ATLAS collaboration
*/

#include "TROOT.h"

void run()
{
  gROOT->ProcessLine(".x init_epara.C+");
  //int dsid=100006;
  
  ////// Photon, E65536, eta 5 10
  //int dsid=100007; // z = 0
  //int dsid=100008; // z = 150
  //int dsid=100009; // z = 50
  //int dsid=100010; // z = -150
  //int dsid=100011; // z = -50
  ////// Photon, E65536, eta 10 15
  //int dsid=100012; // z = 0
  //int dsid=100013; // z = 150
  //int dsid=100014; // z = 50
  //int dsid=100015; // z = -150
  //int dsid=100016; // z = -50
  ////// Photon, E65536, eta 15 20
  //int dsid=100017; // z = 0
  //int dsid=100018; // z = 150
  //int dsid=100019; // z = 50
  //int dsid=100020; // z = -150
  //int dsid=100021; // z = -50
  ////// Photon, E65536, eta 45 50
  //int dsid=100022; // z = 0
  //int dsid=100023; // z = 150
  //int dsid=100024; // z = 50
  //int dsid=100025; // z = -150
  //int dsid=100026; // z = -50
  ////// Photon, E65536, eta 95 100
  //int dsid=100027; // z = 0
  //int dsid=100028; // z = 150
  //int dsid=100029; // z = 50
  //int dsid=100030; // z = -150
  //int dsid=100031; // z = -50
  
  
  TString db="mydb.txt";
  
  for(int dsid=100032;dsid<=100038;dsid++){
    TString parameters = Form("%i,\"%s\"",dsid,db.Data()); 
  
    gROOT->ProcessLine(".x run_epara.C+("+parameters + ")"); 
    
  }
  
  
}

