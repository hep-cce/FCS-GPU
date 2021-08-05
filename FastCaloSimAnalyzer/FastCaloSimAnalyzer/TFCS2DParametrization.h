/*
  Copyright (C) 2002-2017 CERN for the benefit of the ATLAS collaboration
 */

#ifndef TFCS2DParametrization_H
#define TFCS2DParametrization_H

#include "TFCSAnalyzerBase.h"

class TH1F;
class TH2F;

class TFCS2DParametrization: public TFCSAnalyzerBase
{

public:
  TFCS2DParametrization();
  TFCS2DParametrization( TTree*, std::string, std::vector<int> );
  ~TFCS2DParametrization();

  void                CreateShapeHistograms( double cutoff, std::string opt = "" );
  void                PlotShapePolar();
  TH1F*               GetHisto( std::string, int, int, double, std::string opt = "energy" );
  TH2F*               GetParametrization( TH1F* h_radius, int layer, int pca, std::string opt );
  int                 GetNbins( TH1F*, int );
  std::vector<double> CreateEqualEnergyBinning( TH1F*, int );
  std::vector<double> CreateBinning( TH1F* histo );

private:
  int m_debug;

  TTree*           m_tree;
  std::string      m_outputFile;
  std::vector<int> m_vlayer;

  ClassDef( TFCS2DParametrization, 1 );
};

#if defined( __MAKECINT__ )
#  pragma link C++ class TFCS2DParametrization + ;
#endif

#endif
