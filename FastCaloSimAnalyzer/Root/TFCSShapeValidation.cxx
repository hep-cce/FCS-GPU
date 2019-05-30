/*
  Copyright (C) 2002-2017 CERN for the benefit of the ATLAS collaboration
 */

#include "FastCaloSimAnalyzer/TFCSShapeValidation.h"

#include "TString.h"
#include "TMath.h"
#include "TFile.h"
#include "TH1.h"
#include "TH2.h"
#include "TProfile2D.h"
#include "TCanvas.h"

#include "TChain.h"


#include <iostream>
#include <tuple>
#include <map>
#include <algorithm>
#include <fstream>

#include "CLHEP/Random/TRandomEngine.h"

#include "ISF_FastCaloSimEvent/TFCSSimulationState.h"
#include "ISF_FastCaloSimEvent/TFCSTruthState.h"
#include "ISF_FastCaloSimEvent/TFCSExtrapolationState.h"

#include "TFCSSampleDiscovery.h"

TFCSShapeValidation::TFCSShapeValidation(long seed)
{
   m_debug = 0;
   m_geo = 0;
   m_nprint=-1;
   m_firstevent=0;

   m_randEngine = new CLHEP::TRandomEngine();
   m_randEngine->setSeed(seed);
}


TFCSShapeValidation::TFCSShapeValidation(TChain *chain, int layer, long seed)
{
   m_debug = 0;
   m_chain = chain;
   m_output = "";
   m_layer = layer;
   m_geo = 0;
   m_nprint=-1;
   m_firstevent=0;

   m_randEngine = new CLHEP::TRandomEngine();
   m_randEngine->setSeed(seed);
}


TFCSShapeValidation::~TFCSShapeValidation()
{

}

void TFCSShapeValidation::LoadGeo()
{
  if(m_geo) return;

  m_geo = new CaloGeometryFromFile();

  // load geometry files
  m_geo->LoadGeometryFromFile(TFCSSampleDiscovery::geometryName(), TFCSSampleDiscovery::geometryTree(), TFCSSampleDiscovery::geometryMap());
  m_geo->LoadFCalGeometryFromFiles(TFCSSampleDiscovery::geometryNameFCal());
}

void TFCSShapeValidation::LoopEvents(int pcabin=-1)
{
  LoadGeo();

  int nentries = m_nentries;
  int layer = m_layer;
  std::cout << "TFCSShapeValidation::LoopEvents(): Running on layer = " << layer << ", pcabin = " << pcabin << std::endl ;

  InitInputTree(m_chain, layer);

  ///////////////////////////////////
  //// Initialize truth, extraplolation and all validation structures
  ///////////////////////////////////
  m_truthTLV.resize(nentries);
  m_extrapol.resize(nentries);
  
  for(auto& validation : m_validations) {
    std::cout << "========================================================"<<std::endl;
    if(m_debug >= 1) validation.basesim()->setLevel(MSG::DEBUG,true);
    validation.basesim()->set_geometry(m_geo);
#ifdef FCS_DEBUG
    validation.basesim()->Print();
#endif
    validation.simul().reserve(nentries);
    std::cout << "========================================================"<<std::endl<<std::endl;
  }
  
  ///////////////////////////////////
  //// Event loop
  ///////////////////////////////////
  if(m_nprint<0) {
    m_nprint=250;
    if(nentries<5000) m_nprint=100;
    if(nentries<1000) m_nprint=50;
    if(nentries<500) m_nprint=20;
    if(nentries<100) m_nprint=1;
  }
  
  for (int ievent = m_firstevent; ievent < nentries; ievent++)
  //for (int ievent = m_firstevent; ievent < 100; ievent++)
  {
     if (ievent % m_nprint == 0) std::cout << std::endl << "Event: " << ievent << std::endl;
     m_chain->GetEntry(ievent);

     ///////////////////////////////////
     //// Initialize truth
     ///////////////////////////////////
     float px = m_truthPx->at(0);
     float py = m_truthPy->at(0);
     float pz = m_truthPz->at(0);
     float E = m_truthE->at(0);
     int pdgid = m_truthPDGID->at(0);

     TFCSTruthState& truthTLV=m_truthTLV[ievent];
     truthTLV.SetPxPyPzE(px, py, pz, E);
     truthTLV.set_pdgid(pdgid);

     ///////////////////////////////////
     //// OLD, to be removed: should run over all pca bins
     ///////////////////////////////////
     
     if (m_debug >= 1) {
       std::cout << std::endl << "Event: " << ievent ;
       std::cout << " pca = " << pca()<<" m_pca="<< m_pca<<" ";
       truthTLV.Print();
     }

     ///////////////////////////////////
     //// Initialize truth extrapolation to each calo layer
     ///////////////////////////////////
     TFCSExtrapolationState& extrapol=m_extrapol[ievent];
     extrapol.clear();

     float TTC_eta, TTC_phi, TTC_r, TTC_z;

     if (!m_isNewSample)
     {
        TTC_eta = (*m_truthCollection)[0].TTC_entrance_eta[0];
        TTC_phi = (*m_truthCollection)[0].TTC_entrance_phi[0];
        TTC_r = (*m_truthCollection)[0].TTC_entrance_r[0];
        TTC_z = (*m_truthCollection)[0].TTC_entrance_z[0];

        std::cout << std::endl << " TTC size: " << (*m_truthCollection)[0].TTC_entrance_eta.size()<<std::endl;
        
        for(int i=0;i<CaloCell_ID_FCS::MaxSample;++i) {
          if(m_total_layer_cell_energy[i]==0) continue;
		      extrapol.set_OK(i,TFCSExtrapolationState::SUBPOS_ENT, true);
		      extrapol.set_eta(i,TFCSExtrapolationState::SUBPOS_ENT, (*m_truthCollection)[0].TTC_entrance_eta[i]);
		      extrapol.set_phi(i,TFCSExtrapolationState::SUBPOS_ENT, (*m_truthCollection)[0].TTC_entrance_phi[i]);
		      extrapol.set_r(i,TFCSExtrapolationState::SUBPOS_ENT, (*m_truthCollection)[0].TTC_entrance_r[i]);
		      extrapol.set_z(i,TFCSExtrapolationState::SUBPOS_ENT, (*m_truthCollection)[0].TTC_entrance_z[i]);

		      extrapol.set_OK(i,TFCSExtrapolationState::SUBPOS_EXT, true);
		      extrapol.set_eta(i,TFCSExtrapolationState::SUBPOS_EXT, (*m_truthCollection)[0].TTC_back_eta[i]);
		      extrapol.set_phi(i,TFCSExtrapolationState::SUBPOS_EXT, (*m_truthCollection)[0].TTC_back_phi[i]);
		      extrapol.set_r(i,TFCSExtrapolationState::SUBPOS_EXT, (*m_truthCollection)[0].TTC_back_r[i]);
		      extrapol.set_z(i,TFCSExtrapolationState::SUBPOS_EXT, (*m_truthCollection)[0].TTC_back_z[i]);

		      //extrapol.set_OK(i,TFCSExtrapolationState::SUBPOS_MID, true);
		      //extrapol.set_eta(i,TFCSExtrapolationState::SUBPOS_MID, m_TTC_mid_eta->at(0).at(i));
		      //extrapol.set_phi(i,TFCSExtrapolationState::SUBPOS_MID, m_TTC_mid_phi->at(0).at(i));
		      //extrapol.set_r(i,TFCSExtrapolationState::SUBPOS_MID, m_TTC_mid_r->at(0).at(i));
		      //extrapol.set_z(i,TFCSExtrapolationState::SUBPOS_MID, m_TTC_mid_z->at(0).at(i));
		      extrapol.set_OK(i,TFCSExtrapolationState::SUBPOS_MID, true);
		      extrapol.set_eta(i,TFCSExtrapolationState::SUBPOS_MID, 0.5*((*m_truthCollection)[0].TTC_entrance_eta[i] + (*m_truthCollection)[0].TTC_back_eta[i]));
		      extrapol.set_phi(i,TFCSExtrapolationState::SUBPOS_MID, 0.5*((*m_truthCollection)[0].TTC_entrance_phi[i] + (*m_truthCollection)[0].TTC_back_phi[i]));
		      extrapol.set_r  (i,TFCSExtrapolationState::SUBPOS_MID, 0.5*((*m_truthCollection)[0].TTC_entrance_r[i] + (*m_truthCollection)[0].TTC_back_r[i]));
		      extrapol.set_z  (i,TFCSExtrapolationState::SUBPOS_MID, 0.5*((*m_truthCollection)[0].TTC_entrance_z[i] + (*m_truthCollection)[0].TTC_back_z[i]));
		    }
     } else {
        if(m_TTC_IDCaloBoundary_eta->size()>0) {
          extrapol.set_IDCaloBoundary_eta(m_TTC_IDCaloBoundary_eta->at(0));
          extrapol.set_IDCaloBoundary_phi(m_TTC_IDCaloBoundary_phi->at(0));
          extrapol.set_IDCaloBoundary_r(m_TTC_IDCaloBoundary_r->at(0));
          extrapol.set_IDCaloBoundary_z(m_TTC_IDCaloBoundary_z->at(0));
        }

		    TTC_eta = ((*m_TTC_entrance_eta).at(0).at(layer) + (*m_TTC_back_eta).at(0).at(layer) ) / 2 ;

		    TTC_phi = ((*m_TTC_entrance_phi).at(0).at(layer) + (*m_TTC_back_phi).at(0).at(layer)) / 2 ;
		    TTC_r = ((*m_TTC_entrance_r).at(0).at(layer) + (*m_TTC_back_r).at(0).at(layer) ) / 2 ;
		    TTC_z = ((*m_TTC_entrance_z).at(0).at(layer) + (*m_TTC_back_z).at(0).at(layer) ) / 2 ;
		
        for(int i=0;i<CaloCell_ID_FCS::MaxSample;++i) {
//          if(m_total_layer_cell_energy[i]==0) continue;
		      //extrapol.set_OK(i,TFCSExtrapolationState::SUBPOS_ENT, true);
		      extrapol.set_OK(i,TFCSExtrapolationState::SUBPOS_ENT, m_TTC_entrance_OK->at(0).at(i));
		      extrapol.set_eta(i,TFCSExtrapolationState::SUBPOS_ENT, m_TTC_entrance_eta->at(0).at(i));
		      extrapol.set_phi(i,TFCSExtrapolationState::SUBPOS_ENT, m_TTC_entrance_phi->at(0).at(i));
		      extrapol.set_r(i,TFCSExtrapolationState::SUBPOS_ENT, m_TTC_entrance_r->at(0).at(i));
		      extrapol.set_z(i,TFCSExtrapolationState::SUBPOS_ENT, m_TTC_entrance_z->at(0).at(i));

		      //extrapol.set_OK(i,TFCSExtrapolationState::SUBPOS_EXT, true);
		      extrapol.set_OK(i,TFCSExtrapolationState::SUBPOS_EXT, m_TTC_back_OK->at(0).at(i));
		      extrapol.set_eta(i,TFCSExtrapolationState::SUBPOS_EXT, m_TTC_back_eta->at(0).at(i));
		      extrapol.set_phi(i,TFCSExtrapolationState::SUBPOS_EXT, m_TTC_back_phi->at(0).at(i));
		      extrapol.set_r(i,TFCSExtrapolationState::SUBPOS_EXT, m_TTC_back_r->at(0).at(i));
		      extrapol.set_z(i,TFCSExtrapolationState::SUBPOS_EXT, m_TTC_back_z->at(0).at(i));

		      /*
		      //extrapol.set_OK(i,TFCSExtrapolationState::SUBPOS_MID, true);
		      extrapol.set_OK(i,TFCSExtrapolationState::SUBPOS_MID, m_TTC_mid_OK->at(0).at(i));
		      extrapol.set_eta(i,TFCSExtrapolationState::SUBPOS_MID, m_TTC_mid_eta->at(0).at(i));
		      extrapol.set_phi(i,TFCSExtrapolationState::SUBPOS_MID, m_TTC_mid_phi->at(0).at(i));
		      extrapol.set_r(i,TFCSExtrapolationState::SUBPOS_MID, m_TTC_mid_r->at(0).at(i));
		      extrapol.set_z(i,TFCSExtrapolationState::SUBPOS_MID, m_TTC_mid_z->at(0).at(i));
		      */
		      
		      extrapol.set_OK(i,TFCSExtrapolationState::SUBPOS_MID, (extrapol.OK(i,TFCSExtrapolationState::SUBPOS_ENT) && extrapol.OK(i,TFCSExtrapolationState::SUBPOS_EXT)));
		      extrapol.set_eta(i,TFCSExtrapolationState::SUBPOS_MID, 0.5*(m_TTC_entrance_eta->at(0).at(i)+m_TTC_back_eta->at(0).at(i)));
		      extrapol.set_phi(i,TFCSExtrapolationState::SUBPOS_MID, 0.5*(m_TTC_entrance_phi->at(0).at(i)+m_TTC_back_phi->at(0).at(i)));
		      extrapol.set_r(i,TFCSExtrapolationState::SUBPOS_MID, 0.5*(m_TTC_entrance_r->at(0).at(i)+m_TTC_back_r->at(0).at(i)));
		      extrapol.set_z(i,TFCSExtrapolationState::SUBPOS_MID, 0.5*(m_TTC_entrance_z->at(0).at(i)+m_TTC_back_z->at(0).at(i)));
		      
		    }
     }
     if (m_debug >= 1) extrapol.Print();

     if (m_debug == 2)
        std::cout << "TTC eta, phi, r, z = " << TTC_eta << " , " << TTC_phi<< " , " << TTC_r<< " , " << TTC_z << std::endl;

     if(pcabin>=0) if(pca()!=pcabin) continue;

     ///////////////////////////////////
     //// run simulation chain
     ///////////////////////////////////
     
     for(auto& validation : m_validations) {
       if (m_debug >= 1) {
         std::cout << "Simulate : " << validation.basesim()->GetTitle() <<" event="<<ievent<<" E="<<total_energy()<<" Ebin="<<pca()<<std::endl;
       }

       validation.simul().emplace_back(m_randEngine);
       TFCSSimulationState& chain_simul = validation.simul().back();
       validation.basesim()->simulate(chain_simul,&truthTLV,&extrapol); 
       if (m_debug >= 1) {
         chain_simul.Print();
         std::cout << "End simulate : " << validation.basesim()->GetTitle() <<" event="<<ievent<<std::endl<<std::endl;
       }  
     }
  } // end loop over events
  
  
/*  
  TCanvas* c;
  c=new TCanvas(hist_cellSFvsE->GetName(),hist_cellSFvsE->GetTitle());
  hist_cellSFvsE->Draw();
  c->SaveAs(".png");
  
  c=new TCanvas(hist_cellEvsdxdy_org->GetName(),hist_cellEvsdxdy_org->GetTitle());
  hist_cellEvsdxdy_org->SetMaximum(1);
  hist_cellEvsdxdy_org->SetMinimum(0.00001);
  hist_cellEvsdxdy_org->Draw("colz");
  c->SetLogz(true);
  c->SaveAs(".png");
  
  c=new TCanvas(hist_cellEvsdxdy_sim->GetName(),hist_cellEvsdxdy_sim->GetTitle());
  hist_cellEvsdxdy_sim->SetMaximum(1);
  hist_cellEvsdxdy_sim->SetMinimum(0.00001);
  hist_cellEvsdxdy_sim->Draw("colz");
  c->SetLogz(true);
  c->SaveAs(".png");

  c=new TCanvas(hist_cellEvsdxdy_ratio->GetName(),hist_cellEvsdxdy_ratio->GetTitle());
  hist_cellEvsdxdy_ratio->Draw("colz");
  hist_cellEvsdxdy_ratio->SetMaximum(1.0*8);
  hist_cellEvsdxdy_ratio->SetMinimum(1.0/8);
  c->SetLogz(true);
  c->SaveAs(".png");
*/  
}
