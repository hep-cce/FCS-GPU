/*
  Copyright (C) 2002-2023 CERN for the benefit of the ATLAS collaboration
*/

#include "../../ISF_FastCaloSimEvent/ISF_FastCaloSimEvent/TFCS2DFunctionHistogram.h"
#include "TRandom.h"

void test_TFCS2DFunctionHistogram() 
{
  //DEBUG Shape_id211_E65536_eta_20_25_Ebin1_cs2

  TFile* file=TFile::Open("/eos/atlas/atlascerngroupdisk/proj-simul/InputSamplesLocalProd2017/rel_21_0_62/shapePara/mc16_13TeV.pion.E65536.eta020_025.merged_default_z0.shapepara.root");
  
  file->ls();
  
  TH2F* hist=(TH2F*)file->Get("h_r_alpha_layer2_pca1");
  
  TFCS2DFunctionHistogram m_hist(hist);

  for(int i=0;i<1000000000;++i) {
    if(i%10000000==0) cout<<"Iteration "<<i<<endl;
    float alpha, r, rnd1, rnd2;
    rnd1=gRandom->Rndm();
    rnd2=gRandom->Rndm();
    m_hist.rnd_to_fct(alpha,r,rnd1,rnd2);
    if(TMath::IsNaN(alpha) || TMath::IsNaN(r)) {
      cout<<"  Histo: "<<m_hist.get_HistoBordersx().size()-1<<"*"<<m_hist.get_HistoBordersy().size()-1<<" bins: alpha="<<alpha<<" r="<<r<<" rnd1="<<rnd1<<" rnd2="<<rnd2<<endl;
    }
  }  
}


