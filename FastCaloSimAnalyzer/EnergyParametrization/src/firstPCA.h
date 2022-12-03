/*
  Copyright (C) 2002-2023 CERN for the benefit of the ATLAS collaboration
*/

#ifndef firstPCA_h
#define firstPCA_h

#include "TChain.h"
#include "TreeReader.h"
#include "TH1D.h"

class firstPCA
{
  public:
  	
    firstPCA(TChain*, std::string);
    firstPCA();
    virtual ~firstPCA() {}
    void run();
    std::vector<TH1D*> get_relevantlayers_inputs(std::vector<int> &, TreeReader*);
    std::vector<TH1D*> get_cumul_histos(std::vector<std::string> layer, std::vector<TH1D*>);
    static double get_cumulant(double x, TH1D* h);
    void quantiles(TH1D* h, int nq, double* xq, double* yq);

    void set_cumulativehistobins(int);
    void set_edepositcut(double);
    void set_etacut(double,double);
    void apply_etacut(int);
    void set_pcabinning(int,int);
    
  private:
  	
  	int    m_debuglevel;
  	double m_cut_eta_low;
  	double m_cut_eta_high;
  	int    m_apply_etacut;
  	int    m_nbins1;
  	int    m_nbins2;
  	int    m_numberfinebins;
  	double m_edepositcut;
  	std::string m_outfilename;
  	TChain* m_chain;
  	
  ClassDef(firstPCA,1);
  
};

#if defined(__ROOTCLING__) && defined(__FastCaloSimStandAlone__)
#pragma link C++ class firstPCA+;
#endif

#endif
