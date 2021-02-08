/*
  Copyright (C) 2002-2017 CERN for the benefit of the ATLAS collaboration
 */

#ifndef TFCSInputValidationPlots_H
#define TFCSInputValidationPlots_H

#include "TFCSAnalyzerBase.h"

#include <map>

class TFCSInputValidationPlots : public TFCSAnalyzerBase {

public:
  struct binStruct {
    int    nbins;
    double min;
    double max;
  };

public:
  TFCSInputValidationPlots();
  TFCSInputValidationPlots( TTree*, std::string, std::vector<int> );
  ~TFCSInputValidationPlots();

  void PlotTH1( std::string var, std::string xlabel );
  void PlotTH1Layer( std::string var, std::string xlabel );
  void PlotTH1Layer( std::string var, int nbins, double xmin, double xmax, std::string );
  void PlotTH1PCA( std::string var, std::string xlabel );
  void PlotTH1PCA( std::string var, int layer, int nbins, double xmin, double xmax, std::string xlabel );

  void PlotTH2( std::string var, std::string xlabel, std::string ylabel );
  void PlotTH2( std::string var, int layer, int pca, int nbinsx, double xmin, double xmax, int nbinsy, double ymin,
                double ymax, std::string xlabel, std::string ylabel );

  void                CreateBinning( double cutoff );
  std::vector<double> GetEnergyRmax( int layer, int pca );
  double              GetRmax( int layer, int pca, std::string opt = "" );
  double              GetEnergy( int layer, int pca );
  double              GetMaxRmax( std::string opt = "" );
  double              GetMinRmax( std::string opt = "" );
  binStruct           GetBinValues( std::string var, double rmax );

  void CreateInputValidationHTML( std::string filename, std::vector<std::string> );

  void PlotJOValidation( std::vector<std::string>* files, std::string var, std::string xlabel );

private:
  int m_debug;

  TTree*           m_tree;
  std::string      m_outputDir;
  std::vector<int> m_vlayer;

  std::map<std::pair<int, int>, std::vector<double>> EnergyRmax;

  ClassDef( TFCSInputValidationPlots, 1 );
};

#if defined( __MAKECINT__ )
#  pragma link C++ class TFCSInputValidationPlots + ;
#endif

#endif
