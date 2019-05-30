/*
  Copyright (C) 2002-2017 CERN for the benefit of the ATLAS collaboration
*/

{
  // Needs athena environment setup and ISF_FastCaloSimParametrization package compiled.
  // Uses the root interface library created in the compilation of ISF_FastCaloSimParametrization
  gSystem->Load("libISF_FastCaloSimParametrizationLib.so");
  
  TString filename="ExampleEnergyParam.root";
  TFile* fin=TFile::Open(filename);
  if(fin) {
    cout<<"reading from file "<<filename<<endl;
    TFCSParametrizationBase* tin=(TFCSParametrizationBase*)fin->Get("ExampleEnergyParam_id11_E20000_eta02");
    tin->Print();

    TFCSLateralShapeParametrization* sin=(TFCSLateralShapeParametrization*)fin->Get("ExampleShapeParam_id11_E20000_eta02_Ebin1_cs2");
    sin->Print();
  } else {
    cout<<"Creating parametrization object"<<endl;
    TFCSPCAEnergyParametrization t("ExampleEnergyParam_id11_E20000_eta02","Energy parametrization |pdgid|=11 E=20000, eta=0.2");
    t.add_pdgid(11);
    t.add_pdgid(-11);

    t.set_Ekin_nominal(2000);
    t.set_Ekin_min(10000);
    t.set_Ekin_max(50000);

    t.set_eta_nominal(0.225);
    t.set_eta_min(0.20);
    t.set_eta_max(0.25);
    //Would add real content of energy parametrization here, not only the energy and eta range
    
    t.Print();
    
    TFCSNNLateralShapeParametrization s("ExampleShapeParam_id11_E20000_eta02_Ebin1_cs2","Energy parametrization |pdgid|=11 E=20000, eta=0.2, Ebin=1, calosample=2");
    s.add_pdgid(11);
    s.add_pdgid(-11);

    s.set_Ekin_nominal(2000);
    s.set_Ekin_min(10000);
    s.set_Ekin_max(50000);

    s.set_eta_nominal(0.225);
    s.set_eta_min(0.20);
    s.set_eta_max(0.25);
    
    s.set_Ekin_bin(1);

    s.set_calosample(2);
    //Would add real content of shape parametrization here, not only the energy and eta range
    
    s.Print();

    ////////////////////////////
    // Now writing to root file 
    ////////////////////////////

    cout<<"writing to file "<<filename<<endl;
    TFile* fout=TFile::Open(filename,"recreate");
    t.Write();
    s.Write();
    fout->ls();
    fout->Close();
  }  
}

