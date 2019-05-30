#ifndef TFCSMakeFirstPCA_h
#define TFCSMakeFirstPCA_h

#include "TChain.h"
#include "TreeReader.h"
#include "TH1D.h"

namespace CLHEP
{
  class HepRandomEngine;
}

class TFCSMakeFirstPCA
{
  public:
  	
    TFCSMakeFirstPCA(TChain*, std::string);
    TFCSMakeFirstPCA();
    virtual ~TFCSMakeFirstPCA() {};
    //void run();
    void run(CLHEP::HepRandomEngine *randEngine);
    std::vector<int> get_relevantlayers(TreeReader*, double);
    std::vector<TH1D*> get_G4_histos_from_tree(std::vector<int> layer_number, TreeReader* read_inputTree);
    std::vector<TH1D*> get_cumul_histos(std::vector<std::string> layer, std::vector<TH1D*>);
    double get_cumulant(double x, TH1D* h);
    //double get_cumulant_random(double x, TH1D* h);
    double get_cumulant_random(CLHEP::HepRandomEngine *randEngine, double x, TH1D* h);
    double get_edepositcut() {return m_edepositcut;};

    void set_cumulativehistobins(int);
    void set_edepositcut(double);
    void set_etacut(double,double);
    void apply_etacut(int);
    
    void set_dorescale(int flag) {m_dorescale=flag;}
    void use_absolute_layercut(int flag) {m_use_absolute_layercut=flag;}
    
  private:
  	
    int     m_use_absolute_layercut;
    int     m_dorescale;
  	double  m_cut_eta_low;
  	double  m_cut_eta_high;
  	int     m_apply_etacut;
  	int     m_numberfinebins;
  	double  m_edepositcut;
  	std::string m_outfilename;
  	TChain* m_chain;
  	
  ClassDef(TFCSMakeFirstPCA,1);
  
};

#if defined(__ROOTCLING__) && defined(__FastCaloSimStandAlone__)
#pragma link C++ class TFCSMakeFirstPCA+;
#endif

#endif
