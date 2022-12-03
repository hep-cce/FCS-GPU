/*
  Copyright (C) 2002-2023 CERN for the benefit of the ATLAS collaboration
*/

#ifndef secondPCA_h
#define secondPCA_h

#include "TreeReader.h"
#include "TFCSApplyFirstPCA.h"

namespace CLHEP
{
  class HepRandomEngine;
}

class secondPCA: public TFCSApplyFirstPCA
{
  public:
  
    secondPCA(std::string, std::string);
    //void run();
    void run(CLHEP::HepRandomEngine *);
    std::vector<int> getLayerBins(TFile* file, int &bins);
    //void do_pca(vector<string>, int, TreeReader*, int*);
    void do_pca(CLHEP::HepRandomEngine *, std::vector<std::string>, int, TreeReader*, int*);
    std::vector<TH1D*> get_histos_data(std::vector<std::string> layer, TreeReader*);
    double get_lowerBound(TH1D* h_cumulative);
    //double get_cumulant_random(double x, TH1D* h);
    double get_cumulant_random(CLHEP::HepRandomEngine *randEngine, double x, TH1D* h);
    void set_cumulativehistobins(int);
    void set_storeDetails(int);
    void set_skip_regression(int);
    void set_PCAbin(int);
    void set_cut_maxdeviation_regression(double val);
    void set_cut_maxdeviation_smartrebin(double val);
    void set_Ntoys(int val);
    void set_neurons_iteration(int start,int end);
  
  private:
    
    int m_numberfinebins;
    int m_storeDetails;
    int m_PCAbin;
    int m_skip_regression;
    std::string m_outfilename,m_firstpcafilename;
    int m_neurons_start,m_neurons_end,m_ntoys;
    double m_maxdev_regression,m_maxdev_smartrebin;
    
  ClassDef(secondPCA,2);
  
};

#if defined(__ROOTCLING__) && defined(__FastCaloSimStandAlone__)
#pragma link C++ class secondPCA+;
#endif

#endif
