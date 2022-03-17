/*
  Copyright (C) 2002-2017 CERN for the benefit of the ATLAS collaboration
*/
#include "CLHEP/Random/TRandomEngine.h"

#include "TFile.h"
#include "TString.h"
#include "TRandom.h"
#include "TH2.h"
#include "TCanvas.h"
#include "TVector3.h"

#include <iostream>
#include <string>
#include <stdlib.h>

#include "CaloGeometryFromFile.h"
#include "ISF_FastCaloSimEvent/TFCSParametrizationBase.h"
#include "ISF_FastCaloSimEvent/TFCSSimulationState.h"
#include "ISF_FastCaloSimEvent/TFCSTruthState.h"
#include "ISF_FastCaloSimEvent/TFCSExtrapolationState.h"

//#include "runTFCSCreateParametrization.cxx"

void runTFCSEnergyScan(int pdgid = 22, long seed = 42) {
  // init_hit_to_cell_mapping();
  // init_umbers_of_hits();

  CLHEP::TRandomEngine* randEngine = new CLHEP::TRandomEngine();
  randEngine->setSeed(seed);

  TFile* fullchainfile = TFile::Open("TFCSparam_linear_interpolation.root");
  fullchainfile->ls();
  TFCSParametrizationBase* fullchain =
      (TFCSParametrizationBase*)fullchainfile->Get("SelPDGID");
  fullchainfile->Close();

  double etamin = 0.2;
  double etamax = 0.25;

  CaloGeometryFromFile* geo = new CaloGeometryFromFile();
  geo->LoadGeometryFromFile(
      "/afs/cern.ch/atlas/groups/Simulation/FastCaloSimV2/"
      "Geometry-ATLAS-R2-2016-01-00-01.root",
      "ATLAS-R2-2016-01-00-01", "cellId_vs_cellHashId_map.txt");
  TString path_to_fcal_geo_files = "./";
  geo->LoadFCalGeometryFromFiles(
      path_to_fcal_geo_files + "FCal1-electrodes.sorted.HV.09Nov2007.dat",
      path_to_fcal_geo_files + "FCal2-electrodes.sorted.HV.April2011.dat",
      path_to_fcal_geo_files + "FCal3-electrodes.sorted.HV.09Nov2007.dat");
  fullchain->set_geometry(geo);

  // fullchain->Print();

  float Emin = 730;
  float Emax = 92000;
  float logEmin = TMath::Log(Emin);
  float logEmax = TMath::Log(Emax);

  int nxbin = 0;
  double xbin[1000];
  double Ebin = Emin;
  while (Ebin <= Emax) {
    xbin[nxbin] = Ebin;
    ++nxbin;
    Ebin *= 1.1;
  }
  TH2D* response = new TH2D("photon_response", "photon response", nxbin - 1,
                            xbin, 100, 0.5, 1.5);

  for (int ievent = 0; ievent < 1000000; ++ievent) {
    if (ievent % 10000 == 0) cout << "Event " << ievent << endl;
    float logE = logEmin + (logEmax - logEmin) * gRandom->Rndm();
    float E = TMath::Exp(logE);
    float eta = etamin + (etamax - etamin) * gRandom->Rndm();
    float phi = gRandom->Rndm() * 2 * TMath::Pi();
    float M = 0;
    if (pdgid == 11 || pdgid == -11) M = 0.51;
    if (pdgid == 211 || pdgid == -211) M = 139.57;
    float P = TMath::Sqrt(E * E - M * M);
    TVector3 Pvect;
    Pvect.SetPtEtaPhi(1, eta, phi);
    Pvect.SetMag(P);

    TFCSTruthState truthTLV;
    truthTLV.SetVectM(Pvect, M);
    truthTLV.set_pdgid(pdgid);

    // cout<<"E="<<E<<" TLV E="<<truthTLV.E()<<" M="<<truthTLV.M()<<endl;
    // truthTLV.Print();

    TFCSExtrapolationState extrapol;
    extrapol.clear();

    extrapol.set_IDCaloBoundary_eta(eta);
    extrapol.set_IDCaloBoundary_phi(phi);
    extrapol.set_IDCaloBoundary_r(1148);
    extrapol.set_IDCaloBoundary_z(3550);

    for (int i = 0; i < CaloCell_ID_FCS::MaxSample; ++i) {
      extrapol.set_OK(i, TFCSExtrapolationState::SUBPOS_ENT, true);
      extrapol.set_eta(i, TFCSExtrapolationState::SUBPOS_ENT, eta);
      extrapol.set_phi(i, TFCSExtrapolationState::SUBPOS_ENT, phi);
      extrapol.set_r(i, TFCSExtrapolationState::SUBPOS_ENT, geo->rent(i, eta));
      extrapol.set_z(i, TFCSExtrapolationState::SUBPOS_ENT, geo->zent(i, eta));

      extrapol.set_OK(i, TFCSExtrapolationState::SUBPOS_EXT, true);
      extrapol.set_eta(i, TFCSExtrapolationState::SUBPOS_EXT, eta);
      extrapol.set_phi(i, TFCSExtrapolationState::SUBPOS_EXT, phi);
      extrapol.set_r(i, TFCSExtrapolationState::SUBPOS_EXT, geo->rext(i, eta));
      extrapol.set_z(i, TFCSExtrapolationState::SUBPOS_EXT, geo->zext(i, eta));

      extrapol.set_OK(i, TFCSExtrapolationState::SUBPOS_MID, true);
      extrapol.set_eta(i, TFCSExtrapolationState::SUBPOS_MID, eta);
      extrapol.set_phi(i, TFCSExtrapolationState::SUBPOS_MID, phi);
      extrapol.set_r(i, TFCSExtrapolationState::SUBPOS_MID, geo->rmid(i, eta));
      extrapol.set_z(i, TFCSExtrapolationState::SUBPOS_MID, geo->zmid(i, eta));
    }
    // extrapol.Print();

    TFCSSimulationState simul(randEngine);
    fullchain->simulate(simul, &truthTLV, &extrapol);
    // simul.Print();

    response->Fill(E, simul.E() / E);
  }
  TCanvas* c = new TCanvas("photon_response_linear_interpolation",
                           "photon response with linear interpolation");
  response->Draw("colz");
  c->SetLogx();
  c->SaveAs(".png");
  c->SaveAs(".pdf");
}
