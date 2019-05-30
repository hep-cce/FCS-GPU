/*
  Copyright (C) 2002-2018 CERN for the benefit of the ATLAS collaboration
*/

#include "CLHEP/Random/RandGauss.h"

#include "ISF_FastCaloSimEvent/TFCSPCAEnergyParametrization.h"
#include "ISF_FastCaloSimEvent/FastCaloSim_CaloCell_ID.h"

#include "ISF_FastCaloSimEvent/TFCSSimulationState.h"
#include "ISF_FastCaloSimEvent/TFCSExtrapolationState.h"

#include "TFile.h"
#include "TKey.h"
#include "TClass.h"
#include "TMatrixD.h"
#include "TMatrixDSymEigen.h"
#include "TMath.h"

//=============================================
//======= TFCSPCAEnergyParametrization =========
//=============================================

TFCSPCAEnergyParametrization::TFCSPCAEnergyParametrization(const char* name, const char* title):TFCSEnergyParametrization(name,title)
{
  m_numberpcabins=1;
  do_rescale=1;
}

bool TFCSPCAEnergyParametrization::is_match_Ekin_bin(int Ekin_bin) const 
{
  if(Ekin_bin>=1 && Ekin_bin<=n_bins()) return true;
  return false;
}
  
bool TFCSPCAEnergyParametrization::is_match_calosample(int calosample) const 
{
  for(unsigned int i=0;i<m_RelevantLayers.size();i++) {
    if(m_RelevantLayers[i]==calosample) return true;
  }  
  return false;
}

void TFCSPCAEnergyParametrization::Print(Option_t *option) const
{
  TString opt(option);
  bool shortprint=opt.Index("short")>=0;
  bool longprint=msgLvl(MSG::DEBUG) || (msgLvl(MSG::INFO) && !shortprint);
  TString optprint=opt;optprint.ReplaceAll("short","");
  TFCSEnergyParametrization::Print(option);
  
  if(longprint) {
    ATH_MSG(INFO) << optprint <<"  #bins="<<m_numberpcabins<<", layers=";
    for(unsigned int i=0;i<m_RelevantLayers.size();i++) {
      if(i>0) msg()<<", ";
      msg()<<m_RelevantLayers[i];
    }  
    msg()<<endmsg;
  }  
}

FCSReturnCode TFCSPCAEnergyParametrization::simulate(TFCSSimulationState& simulstate,const TFCSTruthState* /*truth*/, const TFCSExtrapolationState* /*extrapol*/)
{
  
  if (!simulstate.randomEngine()) {
    return FCSFatal;
  }

  int pcabin=simulstate.Ebin();
  
  if(pcabin==0)
  {
   simulstate.set_E(0);
   for(int s=0;s<CaloCell_ID_FCS::MaxSample;s++)
   {
    simulstate.set_E(s,0.0);
    simulstate.set_Efrac(s,0.0);
   }
  }
  else
  {
   
   TMatrixD* EV            =m_EV[pcabin-1]; 
   TVectorD* MeanValues    =m_MeanValues[pcabin-1];
   TVectorD* SigmaValues   =m_SigmaValues[pcabin-1];
   TVectorD* Gauss_means   =m_Gauss_means[pcabin-1];
   TVectorD* Gauss_rms     =m_Gauss_rms[pcabin-1];
   std::vector<TFCS1DFunction*> cumulative=m_cumulative[pcabin-1];

   std::vector<int> layerNr;
   for(unsigned int i=0;i<m_RelevantLayers.size();i++)
    layerNr.push_back(m_RelevantLayers[i]);
   
   double* vals_gauss_means=(double*)Gauss_means->GetMatrixArray();
   double* vals_gauss_rms  =Gauss_rms->GetMatrixArray();

   double *output_data = new double[layerNr.size()+1];
   double *input_data = new double[layerNr.size()+1];

   for(unsigned int l=0;l<=layerNr.size();l++)
   {
    double mean=vals_gauss_means[l];
    double rms =vals_gauss_rms[l];
    double gauszz = CLHEP::RandGauss::shoot(simulstate.randomEngine(), mean, rms);
    input_data[l]=gauszz;
   }

   P2X(SigmaValues, MeanValues, EV, layerNr.size()+1, input_data, output_data, layerNr.size()+1);

   double *simdata = new double[layerNr.size()+1];
   double sum_fraction=0.0;
   for(unsigned int l=0;l<=layerNr.size();l++)
   {
    double simdata_uniform=(TMath::Erf(output_data[l]/1.414213562)+1)/2.f;
    
    simdata[l]=cumulative[l]->rnd_to_fct(simdata_uniform);
   
    if(l!=layerNr.size()) //sum up the fractions, but not the totalE
     sum_fraction+=simdata[l];
   }
   
   double scalefactor=1.0/sum_fraction;
   if(!do_rescale) scalefactor=1.0;
  
   for(unsigned int l=0;l<layerNr.size();l++)
   {
    simdata[l]*=scalefactor;
   }
   
   double total_energy=simdata[layerNr.size()]*simulstate.E()/Ekin_nominal();
   simulstate.set_E(total_energy);
   ATH_MSG_DEBUG("set E to total_energy="<<total_energy);
  
   for(int s=0;s<CaloCell_ID_FCS::MaxSample;s++)
   {
    double energyfrac=0.0;
    for(unsigned int l=0;l<layerNr.size();l++)
    {
     if(layerNr[l]==s)
      energyfrac=simdata[l];
    }
    simulstate.set_Efrac(s,energyfrac);
    simulstate.set_E(s,energyfrac*total_energy);
    simulstate.set_SF(scalefactor);
   }

   delete [] output_data;
   delete [] input_data;
   delete [] simdata;

  }

  return FCSSuccess;
}

void TFCSPCAEnergyParametrization::P2X(TVectorD* SigmaValues, TVectorD* MeanValues, TMatrixD *EV, int gNVariables, double *p, double *x, int nTest)
{

  double* gSigmaValues  = SigmaValues->GetMatrixArray();
  double* gMeanValues   = MeanValues->GetMatrixArray();
  double* gEigenVectors = EV->GetMatrixArray();

  for(int i = 0; i < gNVariables; i++)
    {
      x[i] = gMeanValues[i];
      for(int j = 0; j < nTest; j++)
        {
          x[i] += p[j] * gSigmaValues[i] * (double)(gEigenVectors[i *  gNVariables + j]);
        }
    }
}

bool TFCSPCAEnergyParametrization::loadInputs(TFile* file)
{
  return loadInputs(file, "");
}

bool TFCSPCAEnergyParametrization::loadInputs(TFile* file, std::string folder)
{
  
  bool load_ok=1;
  
  int trynext=1;
  TString x;
  if(folder=="") x="bin";
   else x=folder+"/bin";
  while(trynext)
    {
      IntArray* test  =(IntArray*)file->Get(x+Form("%i/pca/RelevantLayers",m_numberpcabins));
      if(test)
        {
          m_numberpcabins++;
          delete test;
        }
      else
        trynext=0;
    }
  m_numberpcabins-=1;

  file->cd(x+"1/pca");
  IntArray* RelevantLayers=(IntArray*)gDirectory->Get("RelevantLayers");
  if(RelevantLayers == NULL)
  {
    ATH_MSG_ERROR("TFCSPCAEnergyParametrization::m_RelevantLayers in first pcabin is null!");
    load_ok=false;
  }
  
  if(!load_ok) return false;
  
  m_RelevantLayers.reserve(RelevantLayers->GetSize());
  for(int i=0;i<RelevantLayers->GetSize();i++) m_RelevantLayers.push_back(RelevantLayers->GetAt(i));
    
  for(int bin=1;bin<=m_numberpcabins;bin++)
    {

      file->cd(x+Form("%i/pca",bin));

      TMatrixDSym* symCov     =(TMatrixDSym*)gDirectory->Get("symCov");
      TVectorD* MeanValues    =(TVectorD*)gDirectory->Get("MeanValues");
      TVectorD* SigmaValues   =(TVectorD*)gDirectory->Get("SigmaValues");
      TVectorD* Gauss_means   =(TVectorD*)gDirectory->Get("Gauss_means");
      TVectorD* Gauss_rms     =(TVectorD*)gDirectory->Get("Gauss_rms");
      
      if(symCov == NULL)       {ATH_MSG_WARNING("TFCSPCAEnergyParametrization::symCov in pcabin "<<bin<<" is null!"); load_ok=false;}
      if(MeanValues == NULL)   {ATH_MSG_WARNING("TFCSPCAEnergyParametrization::MeanValues in pcabin "<<bin<<" is null!"); load_ok=false;}
      if(SigmaValues == NULL)  {ATH_MSG_WARNING("TFCSPCAEnergyParametrization::SigmaValues in pcabin "<<bin<<" is null!"); load_ok=false;}
      if(Gauss_means == NULL)  {ATH_MSG_WARNING("TFCSPCAEnergyParametrization::Gauss_means in pcabin "<<bin<<" is null!"); load_ok=false;}
      if(Gauss_rms == NULL)    {ATH_MSG_WARNING("TFCSPCAEnergyParametrization::Gause_rms in pcabin "<<bin<<" is null!"); load_ok=false;}
      
      if(!load_ok) return false;
      
      TMatrixDSymEigen cov_eigen(*symCov);
      TMatrixD *EV = new TMatrixD(cov_eigen.GetEigenVectors());
      m_EV.push_back(EV);
      m_MeanValues.push_back(MeanValues);
      m_SigmaValues.push_back(SigmaValues);
      m_Gauss_means.push_back(Gauss_means);
      m_Gauss_rms.push_back(Gauss_rms);
      
      std::vector<std::string> layer;
      std::vector<int> layerNr;
      
      for(unsigned int i=0;i<m_RelevantLayers.size();i++)
        layerNr.push_back(m_RelevantLayers[i]);

      for(unsigned int i=0;i<layerNr.size();i++)
        {
          std::string thislayer=Form("layer%i",layerNr[i]);
          layer.push_back(thislayer);
        }
      layer.push_back("totalE");

      std::vector<TFCS1DFunction*> cumulative;

      for(unsigned int l=0;l<layer.size();l++)
        {
          file->cd(Form("%s/bin%i/%s",folder.c_str(),bin,layer[l].c_str()));

          TFCS1DFunction* fct;
          fct=(TFCS1DFunction*)gDirectory->Get("TFCS1DFunctionRegression");
          if(!fct)
            fct=(TFCS1DFunction*)gDirectory->Get("TFCS1DFunctionRegressionTF");
          if(!fct)
            fct=(TFCS1DFunction*)gDirectory->Get("TFCS1DFunctionHistogram");
          cumulative.push_back(fct);

        }

      m_cumulative.push_back(cumulative);

   }

 return true;

}

void TFCSPCAEnergyParametrization::clean()
{
 for(unsigned int i=0;i<m_EV.size();i++)
  delete m_EV[i];
}
