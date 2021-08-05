/*
  Copyright (C) 2002-2021 CERN for the benefit of the ATLAS collaboration
*/

#ifndef TFCSApplyFirstPCA_h
#define TFCSApplyFirstPCA_h

#include "TChain.h"
#include "TreeReader.h"
#include "TH1D.h"
#include "TFCSMakeFirstPCA.h"
#include "ISF_FastCaloSimEvent/TFCSSimulationState.h"

namespace CLHEP
{
  class HepRandomEngine;
}


class TFCSApplyFirstPCA: public TFCSMakeFirstPCA
{
public:

  TFCSApplyFirstPCA(){};
  TFCSApplyFirstPCA( std::string MakeFirstPCA_rootfilename );
  virtual ~TFCSApplyFirstPCA(){};

  void init();

  void                set_cumulative_energy_histos( std::vector<TH1D*> cumul_inputdata );
  int                 get_PCAbin_from_simstate( TFCSSimulationState& );
  std::vector<double> get_PCAdata_from_simstate( TFCSSimulationState& );

  // void run_over_chain(TChain* chain, string outfilename);
  void run_over_chain( CLHEP::HepRandomEngine* randEngine, TChain* chain, std::string outfilename );

  void quantiles( TH1D* h, int nq, double* xq, double* yq );
  void set_pcabinning( int, int );
  void print_binning();
  void set_dorescale( int flag ) { m_dorescale = flag; }
  int  get_number_compos() { return m_layer_totE_name.size(); }

private:
  int                              m_dorescale;
  int                              m_nbins1;
  int                              m_nbins2;
  std::string                      m_infilename;
  TPrincipal*                      m_principal;
  std::vector<double>              m_yq;
  std::vector<std::vector<double>> m_yq2d;
  std::vector<int>                 m_layer_number;
  std::vector<std::string>         m_layer_totE_name;
  std::vector<TH1D*>               m_cumulative_energies;

  ClassDef( TFCSApplyFirstPCA, 1 );
};

#if defined( __ROOTCLING__ ) && defined( __FastCaloSimStandAlone__ )
#  pragma link C++ class TFCSApplyFirstPCA + ;
#endif

#endif
