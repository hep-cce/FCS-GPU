/*
  Copyright (C) 2002-2021 CERN for the benefit of the ATLAS collaboration
*/

#ifndef TFCSVertexZPositionStudies_H
#define TFCSVertexZPositionStudies_H

#include <iostream>
#include <map>
#include <string>
// #include <tuple>
// #include <algorithm>
#include "TH1.h"
#include "TH1F.h"
#include "TH2.h"
#include "TH2F.h"
#include "TFCSAnalyzerBase.h"
#include "TFile.h"

using namespace std;

class TFCSVertexZPositionStudies {
public:
  TFCSVertexZPositionStudies();
  ~TFCSVertexZPositionStudies();

  void loadFiles( string dirname, string filename_nominal, vector<string>& filenames_shifted );
  void initializeLayersAndPCAs();

  inline vector<int> getLayers() { return m_layers; }
  inline vector<int> getPCAs() { return m_pcas; }

  void loadHistogramsInLayerAndPCA( int layer, int pca, string histname );
  void loadHistogramsAllPCAs( int layer, string histname );
  void loadMeanEnergyHistogram( int pca, string histname );
  void loadMeanEnergyHistogram( string histname );
  void loadHistogramsForFixedZVAndLayer( int zv_index, int layer, string histname );

  void checkPCABinsDecomposition( int layer, string histname );

  void normalizeHistograms();
  void deleteHistograms();

  void findBinning( bool useMMbinning = false, double factor = 1. / 3, double quantile = 0.003 );
  void rebinHistos();
  void plotHistograms( string outputDir, bool ratio_plots = false, bool drawErrorBars = false,
                       bool useLogScale = false );
  void plotHistograms( vector<TH1F*>& histos, string outputDir, bool drawErrorBars = false, bool useLogScale = false );
  void printMeanValues();

  inline void    setName( const TString name ) { m_name = name; }
  inline TString getName() { return m_name; }
  inline void    setXTitle( const TString xtitle ) { m_xtitle = xtitle; }
  inline void    setYTitle( const TString ytitle ) { m_ytitle = ytitle; }

  void setParticleInfo( string filename ); // Function will attempt to determine particle info from filename

private:
  TH1F*          m_hist_nominal;
  vector<TH1F*>  m_histos;
  vector<TH1F*>  m_ratios;
  TFile*         m_file_nominal;
  vector<TFile*> m_files_shifted;
  vector<string> m_map_indexes;
  vector<int>    m_pcas;
  vector<int>    m_layers;

  vector<string> m_legendNames;
  vector<string> m_vertexZPositions;
  string         m_info;
  string         m_particle_eta;
  string         m_particle_energy;
  string         m_particle_type;

  vector<double> m_binning;

  string m_name;
  string m_dirname;
  int    m_nshifted;

  int m_currentPCA;
  int m_currentLayer;

  TString m_xtitle;
  TString m_ytitle;
};

#if defined( __MAKECINT__ )
#  pragma link C++ class TFCSVertexZPositionStudies + ;
#endif

#endif
