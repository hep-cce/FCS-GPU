/*
  Copyright (C) 2002-2017 CERN for the benefit of the ATLAS collaboration
 */

#ifndef TFCSAnalyzerBase_H
#define TFCSAnalyzerBase_H

#include "FCS_Cell.h"
#include <map>

class TH1;
class TH2;
class TH1F;
class TH2F;
class TH1D;
class TCanvas;
class TProfile;
class TProfile2D;
class TChain;
class TTree;




class TFCSAnalyzerBase
{
public:


   TFCSAnalyzerBase();
   virtual ~TFCSAnalyzerBase();


   struct CaloCell
   {
      float r;
      float z;
      float eta;
      float deta;
      float dphi;

   };



   TH1F* InitTH1(std::string histname, std::string histtype, int nbins, float low, float high, std::string xtitle = "", std::string ytitle = "");
   TH2F* InitTH2(std::string histname, std::string histtype, int nbinsx, float lowx, float highx, int nbinsy, float lowy, float highy, std::string xtitle = "", std::string ytitle = "");
   TProfile* InitTProfile1D(std::string histname, std::string histtype, int nbinsx, float lowx, float highx, std::string xtitle = "", std::string ytitle = "", std::string profiletype = "S");
   TProfile2D* InitTProfile2D(std::string histname, std::string histtype, int nbinsx, float lowx, float highx, int nbinsy, float lowy, float highy, std::string xtitle = "", std::string ytitle = "", std::string profiletype = "S");
   static void Fill(TH1 *h, float value, float weight);
   static void Fill(TH2 *h, float valuex, float valuey, float weight);
   static void Fill(TProfile *h, float valuex, float valuey, float weight);
   static void Fill(TProfile2D *h, float valuex, float valuey, float valuez, float weight);

   static void autozoom(TH1* h1, double &min, double &max, double &rmin, double &rmax);
   static TH1D* refill(TH1* h_in, double min, double max, double rmin, double rmax);

   static void GetTH1TTreeDraw(TH1F*& histo, TTree* tree, std::string var, std::string* cut, int nbins, double xmin, double xmax);
   void GetTH2TTreeDraw(TH2F*& hist, TTree* tree, std::string var, std::string* cut, int nbinsx, double xmin, double xmax, int nbinsy, double ymin, double ymax);


   static TCanvas* PlotTH1Ratio(TH1F* h1, TH1F* h2, std::string label, std::string xlabel, std::string leg1, std::string leg2, std::string ylabel1, std::string ylabel2 );
   static TCanvas* PlotPolar(TH2F* h, std::string label, std::string xlabel, std::string ylabel, std::string zlabel, int zoom_level = 1);


   std::tuple<float, float> GetUnitsmm(float eta_hits, float deta, float dphi, CaloCell* cell);
   static std::tuple<float, float> GetUnitsmm(float eta_hits, float deta, float dphi, float cell_r, float cell_z);

   void InitInputTree(TChain*, int);

   static double GetParticleMass(int pdgid);
   static double Mom2Etot(double mass, double mom);
   static double Mom2Etot(int pdgid, double mom);  
   static double Mom2Ekin(int pdgid, double mom);  
   static double Mom2Ekin_min(int pdgid, double mom);
   static double Mom2Ekin_max(int pdgid, double mom); 
   static float DeltaPhi(float, float);
   std::vector<float> Getxbins(TH1F *histo, int nbins);
   static double GetBinUpEdge(TH1F* histo, float cutoff);
   bool findWord(const std::string sentence, std::string search);
   std::string replaceChar(std::string str, char find, char replace);


   void MakeColorVector();
   TString GetLabel();
   TString GetLayerName(int layerid);
   void CreateHTML(std::string filename, std::vector<std::string> histNames);



   void set_Debug(int debug_) { m_debug = debug_; }
   void set_IsNewSample(bool newsample_) { m_isNewSample = newsample_; }
   int  get_Nentries() const {return m_nentries; };
   void set_Nentries(int nentries_) {m_nentries = nentries_; }
   void set_label(std::string label_) { m_label = label_; }
   void set_merge(std::string merge_) {m_merge = merge_; }
   void set_particle(std::string particle_) { m_particle = particle_; }
   void set_energy(float energy_) { m_energy = energy_; }
   void set_eta(float etamin_, float etamax_) { m_etamin = etamin_; m_etamax = etamax_; }

   int pca() const {return m_pca;};
   const double& total_energy() const {return m_total_energy;};
   const std::vector<double>& total_layer_cell_energy() const {return m_total_layer_cell_energy;};
   FCS_matchedcellvector* cellVector() const {return m_cellVector;};
   FCS_matchedcellvector* avgcellVector() const {return m_avgcellVector;};

   std::map< std::string , TH1* >& histMap() {return m_histMap;};

protected:

   int m_debug;
   bool m_isNewSample;
   int m_nentries;
   std::string m_label;
   std::string m_merge;
   std::string m_particle;
   float m_energy;
   float m_etamin;
   float m_etamax;





   std::vector<Color_t> v_color;

   std::vector<std::string> histNameVec;
   std::vector<TH1*> histVec;
   std::map< std::string , TH1* > m_histMap;

   // * reading input TTree

   int m_pca;
   float m_total_hit_energy;
   float m_total_cell_energy;
   std::vector<double> m_total_layer_cell_energy;
   double m_total_energy;

   FCS_matchedcellvector *m_cellVector;
   FCS_matchedcellvector *m_avgcellVector;
   std::vector<FCS_truth> *m_truthCollection;
   std::vector<float> *m_truthPx;
   std::vector<float> *m_truthPy;
   std::vector<float> *m_truthPz;
   std::vector<float> *m_truthE;
   std::vector<int> *m_truthPDGID;
   std::vector<std::vector<bool>>  *m_TTC_entrance_OK;
   std::vector<std::vector<float>> *m_TTC_entrance_eta;
   std::vector<std::vector<float>> *m_TTC_entrance_phi;
   std::vector<std::vector<float>> *m_TTC_entrance_r;
   std::vector<std::vector<float>> *m_TTC_entrance_z;
   std::vector<std::vector<bool>>  *m_TTC_mid_OK;
   std::vector<std::vector<float>> *m_TTC_mid_eta;
   std::vector<std::vector<float>> *m_TTC_mid_phi;
   std::vector<std::vector<float>> *m_TTC_mid_r;
   std::vector<std::vector<float>> *m_TTC_mid_z;
   std::vector<std::vector<bool>>  *m_TTC_back_OK;
   std::vector<std::vector<float>> *m_TTC_back_eta;
   std::vector<std::vector<float>> *m_TTC_back_phi;
   std::vector<std::vector<float>> *m_TTC_back_r;
   std::vector<std::vector<float>> *m_TTC_back_z;
   std::vector<float> *m_TTC_IDCaloBoundary_eta;
   std::vector<float> *m_TTC_IDCaloBoundary_phi;
   std::vector<float> *m_TTC_IDCaloBoundary_r;
   std::vector<float> *m_TTC_IDCaloBoundary_z;

   ClassDef(TFCSAnalyzerBase, 1);
};

#if defined(__MAKECINT__)
#pragma link C++ class TFCSAnalyzerBase+;
#endif

#endif
