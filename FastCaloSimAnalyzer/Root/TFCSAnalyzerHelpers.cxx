/*
  Copyright (C) 2002-2018 CERN for the benefit of the ATLAS collaboration
*/

#include <TFile.h>
#include <TGraph.h>
#include <TMath.h>
#include <TSystem.h>

#include "FastCaloSimAnalyzer/TFCSHistoLateralShapeParametrization.h"
#include "FastCaloSimAnalyzer/TFCSHistoLateralShapeParametrizationFCal.h"
#include "FastCaloSimAnalyzer/TFCSHitCellMappingWiggle.h"
#include "ISF_FastCaloSimEvent/TFCS1DFunctionInt32Histogram.h"
#include "ISF_FastCaloSimEvent/TFCSCenterPositionCalculation.h"
#include "ISF_FastCaloSimEvent/TFCSEnergyBinParametrization.h"
#include "ISF_FastCaloSimEvent/TFCSEnergyInterpolationLinear.h"
#include "ISF_FastCaloSimEvent/TFCSEnergyInterpolationSpline.h"
#include "ISF_FastCaloSimEvent/TFCSHitCellMappingFCal.h"
#include "ISF_FastCaloSimEvent/TFCSLateralShapeParametrizationHitBase.h"
#include "ISF_FastCaloSimEvent/TFCSLateralShapeParametrizationHitNumberFromE.h"
#include "ISF_FastCaloSimEvent/TFCSPCAEnergyParametrization.h"
#include "ISF_FastCaloSimEvent/TFCSParametrizationEbinChain.h"
#include "ISF_FastCaloSimEvent/TFCSParametrizationEkinSelectChain.h"
#include "ISF_FastCaloSimParametrization/MeanAndRMS.h"

#include "FastCaloSimAnalyzer/TFCSAnalyzerBase.h"
#include "FastCaloSimAnalyzer/TFCSAnalyzerHelpers.h"
#include "FastCaloSimAnalyzer/TFCSLateralShapeParametrizationHitChain.h"

#include "TFCSSampleDiscovery.h"

namespace FCS {

  void init_hit_to_cell_mapping( LateralShapeParametrizationArray& mapping, bool isNewWiggle ) {
    if ( isNewWiggle ) {
      init_hit_to_cell_mapping_with_wiggle( mapping, 1, 0, 1.5 );
      init_hit_to_cell_mapping_with_wiggle( mapping, 2, 0, 1.5 );
      init_hit_to_cell_mapping_with_wiggle( mapping, 3, 0, 1.35 );

      init_hit_to_cell_mapping_with_wiggle( mapping, 5, 1.35, 2.60 );
      init_hit_to_cell_mapping_with_wiggle( mapping, 6, 1.35, 3.45 );
      init_hit_to_cell_mapping_with_wiggle( mapping, 7, 1.5, 3.35 );
    } else {
      init_hit_to_cell_mapping_with_wiggle( mapping, 1, {"eta_020_025", "eta_100_105"}, {0, 0.6, 100} );
      init_hit_to_cell_mapping_with_wiggle( mapping, 2, {"eta_020_025", "eta_100_105"}, {0, 0.6, 100} );
      init_hit_to_cell_mapping_with_wiggle( mapping, 3, {"eta_020_025", "eta_100_105"}, {0, 0.6, 100} );

      init_hit_to_cell_mapping_with_wiggle( mapping, 5, {"eta_200_205"}, {0, 100} );
      init_hit_to_cell_mapping_with_wiggle( mapping, 6, {"eta_200_205", "eta_280_285"}, {0, 2.5, 100} );
      init_hit_to_cell_mapping_with_wiggle( mapping, 7, {"eta_200_205", "eta_280_285"}, {0, 2.5, 100} );
    }

    for ( int ilayer = 0; ilayer < 24; ++ilayer ) {
      if ( mapping[ilayer] == 0 ) {

        if ( ilayer < 21 ) {
          mapping[ilayer] = new TFCSHitCellMapping( Form( "hit_to_cell_mapping_%d", ilayer ),
                                                    Form( "hit to cell mapping sampling %d", ilayer ) );
        } else {
          mapping[ilayer] = new TFCSHitCellMappingFCal( Form( "hit_to_cell_mapping_%d", ilayer ),
                                                        Form( "hit to cell mapping sampling %d", ilayer ) );
        }

        mapping[ilayer]->set_calosample( ilayer );
      }
      std::cout << "Wiggle for sampling " << ilayer << std::endl;
#ifdef FCS_DEBUG
      mapping[ilayer]->Print();
#endif
    }
  }

  void init_hit_to_cell_mapping_with_wiggle( LateralShapeParametrizationArray& mapping, int sampling,
                                             const std::vector<std::string>& etaRange,
                                             const std::vector<float>&       etaLowEdge,
                                             const std::vector<float>&       cellDphiHalve ) {
    TFCSHitCellMappingWiggle* wigglefunc =
        new TFCSHitCellMappingWiggle( Form( "hit_to_cell_mapping_wiggle_%d", sampling ),
                                      Form( "hit to cell mapping with wiggle sampling %d", sampling ) );
    wigglefunc->set_calosample( sampling );

    std::vector<const TFCS1DFunction*> functions( etaRange.size() );
    for ( unsigned int ifnc = 0; ifnc < etaRange.size(); ++ifnc ) {
      std::string filename = TFCSSampleDiscovery::getWiggleName( etaRange[ifnc], sampling );
      functions[ifnc]      = nullptr;
      auto wigglefile      = std::unique_ptr<TFile>( TFile::Open( filename.c_str() ) );
      if ( !wigglefile ) {
        std::cerr << "Error: Could not open file '" << filename << "'" << std::endl;
        return;
      }
      if ( wigglefile ) {
        TString histname   = Form( "pos_eff_phi_deriv_Sampling_%d", sampling );
        TH1*    wigglehist = (TH1*)wigglefile->Get( histname );
        if ( wigglehist ) {
          std::cout << "Get wiggle histogram " << histname << " from file:" << filename << std::endl;
          TFCS1DFunctionInt32Histogram* func   = new TFCS1DFunctionInt32Histogram( wigglehist );
          float                         xscale = 1;
          if ( cellDphiHalve.size() == etaRange.size() ) xscale = cellDphiHalve[ifnc];
          for ( auto& ele : func->get_HistoBordersx() ) ele *= xscale;
          functions[ifnc] = func;
        } else {
          std::cout << "Wiggle hist " << histname << " not found, do ls" << std::endl;
#ifdef FCS_DEBUG
          wigglefile->ls();
#endif
        }
        wigglefile->Close();
      } else {
        std::cout << "Wiggle file " << filename << " not found" << std::endl;
      }
    }
    wigglefunc->initialize( functions, etaLowEdge );

    mapping[sampling] = wigglefunc;
  }

  void init_hit_to_cell_mapping_with_wiggle( LateralShapeParametrizationArray& mapping, int sampling, double rangeMin,
                                             double rangeMax, const std::vector<float>& cellDphiHalve ) {
    int                      int_rangemin = TMath::Nint( 100 * rangeMin );
    int                      int_rangemax = TMath::Nint( 100 * rangeMax );
    std::vector<std::string> etarange;
    std::vector<float>       eta_low_edge;
    std::vector<float>       vec_etamin;
    for ( int int_etaMin = int_rangemin; int_etaMin < int_rangemax; int_etaMin += 5 ) {

      int         int_etaMax   = int_etaMin + 5;
      double      etaMin       = int_etaMin / 100.0;
      std::string str_etarange = "eta_" + std::to_string( int_etaMin ) + "_" + std::to_string( int_etaMax );
      etarange.push_back( str_etarange );
      vec_etamin.push_back( etaMin );

      if ( ( sampling == 5 or sampling == 6 or sampling == 7 ) and int_etaMin == int_rangemin )
        eta_low_edge.push_back( 0 );
      else
        eta_low_edge.push_back( etaMin );
    }
    eta_low_edge.push_back( 100 );

    TFCSHitCellMappingWiggle* wigglefunc =
        new TFCSHitCellMappingWiggle( Form( "hit_to_cell_mapping_wiggle_%d", sampling ),
                                      Form( "hit to cell mapping with wiggle sampling %d", sampling ) );
    wigglefunc->set_calosample( sampling );

    std::vector<const TFCS1DFunction*> functions( etarange.size() );
    for ( unsigned int ifnc = 0; ifnc < etarange.size(); ++ifnc ) {
      std::string filename = TFCSSampleDiscovery::getWiggleName( etarange[ifnc], sampling, true );
      functions[ifnc]      = nullptr;
      auto wigglefile      = std::unique_ptr<TFile>( TFile::Open( filename.c_str() ) );
      if ( !wigglefile ) {
        std::cerr << "Error: Could not open file '" << filename << "'" << std::endl;
        return;
      }
      if ( wigglefile ) {
        TString histname = "";
        if ( ( sampling == 6 or sampling == 7 ) and vec_etamin[ifnc] > 2.5 ) {
          std::cout << "In different graunalrity. sampling: " << sampling << " eta: " << vec_etamin[ifnc] << std::endl;
          histname = Form( "cs%d_above_etaboundary_efficiency_derivative_positive", sampling );
        } else {
          histname = Form( "cs%d_efficiency_derivative_positive", sampling );
        }

        TH1* wigglehist = (TH1*)wigglefile->Get( histname );
        if ( wigglehist ) {
          std::cout << "Get wiggle histogram " << histname << " from file:" << filename << std::endl;
          TFCS1DFunctionInt32Histogram* func   = new TFCS1DFunctionInt32Histogram( wigglehist );
          float                         xscale = 1;
          if ( cellDphiHalve.size() == etarange.size() ) xscale = cellDphiHalve[ifnc];
          for ( auto& ele : func->get_HistoBordersx() ) ele *= xscale;
          functions[ifnc] = func;
        } else {
          std::cout << "Wiggle hist " << histname << " not found, do ls" << std::endl;
#ifdef FCS_DEBUG
          wigglefile->ls();
#endif
        }
        wigglefile->Close();
      } else {
        std::cout << "Wiggle file " << filename << " not found" << std::endl;
      }
    }
    wigglefunc->initialize( functions, eta_low_edge );

    mapping[sampling] = wigglefunc;
  }

  void init_numbers_of_hits( LateralShapeParametrizationArray& mapping, float scale ) {
    // scale the stochastic term for fluctuation
    // EM calorimeters
    for ( int ilayer = 0; ilayer < 8; ++ilayer ) {
      if ( mapping[ilayer] == 0 ) {
        mapping[ilayer] = new TFCSLateralShapeParametrizationHitNumberFromE(
            "numbers_of_hits_EM", "Calc numbers of hits EM", 0.101 * scale, 0.002 );
        mapping[ilayer]->set_calosample( ilayer );
      }
    }

    // HEC calorimeters
    for ( int ilayer = 8; ilayer < 12; ++ilayer ) {
      if ( mapping[ilayer] == 0 ) {
        /// https://cds.cern.ch/record/684196
        mapping[ilayer] = new TFCSLateralShapeParametrizationHitNumberFromE( "numbers_of_hits_HEC",
                                                                             "Calc numbers of hits HEC", 0.762, 0.0 );
        mapping[ilayer]->set_calosample( ilayer );
      }
    }

    // Tile calorimeters
    for ( int ilayer = 12; ilayer < 21; ++ilayer ) {
      if ( mapping[ilayer] == 0 ) {
        mapping[ilayer] = new TFCSLateralShapeParametrizationHitNumberFromE(
            "numbers_of_hits_TILE", "Calc numbers of hits TILE", 0.564, 0.055 );
        mapping[ilayer]->set_calosample( ilayer );
      }
    }

    // FCAL calorimeters
    for ( int ilayer = 21; ilayer < 24; ++ilayer ) {
      if ( mapping[ilayer] == 0 ) {
        mapping[ilayer] = new TFCSLateralShapeParametrizationHitNumberFromE(
            "numbers_of_hits_TILE", "Calc numbers of hits TILE", 0.285, 0.035 );
        mapping[ilayer]->set_calosample( ilayer );
      }
    }
  }

  TFCSLateralShapeParametrizationHitBase* NewCenterPositionCalculation( std::string fileName, int pdgId, int intMom,
                                                                        double etaMin, double etaMax, int Ebin,
                                                                        int cs ) {
    int     int_etamin = TMath::Nint( 100 * etaMin );
    int     int_etamax = TMath::Nint( 100 * etaMax );
    TString centerPosParam_name =
        Form( "CenterPosParam_id%d_Mom%d_eta_%d_%d_Ebin%d_cs%d", pdgId, intMom, int_etamin, int_etamax, Ebin, cs );
    TString centerPosParam_title = Form( "Center pos param for id=%d Mom=%d %4.2f<eta<%4.2f Ebin=%d cs=%d", pdgId,
                                         intMom, etaMin, etaMax, Ebin, cs );
    TString shapehist( Form( "h_r_alpha_layer%d_pca%d", cs, Ebin ) );
    auto    extrapfile = std::unique_ptr<TFile>( TFile::Open( fileName.c_str() ) );
    if ( !extrapfile ) {
      std::cerr << "Error: Could not open file '" << fileName << "'" << std::endl;
      throw std::runtime_error( "Error: Could not open file '" + fileName + "'" );
    }

    TMatrixT<float>* tm_mean_weight = (TMatrixT<float>*)extrapfile->Get( "tm_mean_weight" );

    TFCSCenterPositionCalculation* centerPosCalc =
        new TFCSCenterPositionCalculation( centerPosParam_name, centerPosParam_title );

    centerPosCalc->setExtrapWeight( ( *tm_mean_weight )[cs][Ebin] );
    centerPosCalc->set_calosample( cs );
    delete tm_mean_weight;
    return centerPosCalc;
  }

  TFCSLateralShapeParametrizationHitBase* NewHistoShapeParametrization( std::string fileName, int pdgId, int intMom,
                                                                        double etaMin, double etaMax, int Ebin,
                                                                        int cs ) {
    double Ekin     = TFCSAnalyzerBase::Mom2Ekin( pdgId, intMom );
    double Ekin_min = TFCSAnalyzerBase::Mom2Ekin_min( pdgId, intMom );
    double Ekin_max = TFCSAnalyzerBase::Mom2Ekin_max( pdgId, intMom );

    int     int_etaMin = TMath::Nint( 100 * etaMin );
    int     int_etaMax = TMath::Nint( 100 * etaMax );
    TString shapepara_name =
        Form( "Shape_id%d_Mom%d_eta_%d_%d_Ebin%d_cs%d", pdgId, intMom, int_etaMin, int_etaMax, Ebin, cs );
    TString shapepara_title =
        Form( "Shape param for id=%d Mom=%d %4.2f<eta<%4.2f Ebin=%d cs=%d", pdgId, intMom, etaMin, etaMax, Ebin, cs );
    TString shapehist( Form( "h_r_alpha_layer%d_pca%d", cs, Ebin ) );
    auto    shapefile = std::unique_ptr<TFile>( TFile::Open( fileName.c_str() ) );
    if ( !shapefile ) {
      std::cerr << "Error: Could not open file '" << fileName << "'" << std::endl;
      throw std::runtime_error( "Error: Could not open file '" + fileName + "'" );
    }
    // #ifdef FCS_DEBUG
    // shapefile->ls();
    // #endif
    TH2* inhist = (TH2*)shapefile->Get( shapehist );
    if ( !inhist ) {
      TString shapehist2( Form( "cs%d_pca%d_hist_hitenergy_alpha_radius_2D", cs, Ebin ) );
      inhist = (TH2*)shapefile->Get( shapehist2 );
      if ( !inhist ) {
        std::cout << "ERROR: neither histograms '" << shapehist << "' nor '" << shapehist2 << "' found in file "
                  << fileName << std::endl;
        inhist = new TH2D( "dummy", "dummy", 10, 0, 100, 6, -3, 3 );
        inhist->Fill( 0.0, 0.0 );
        // return nullptr;
      }
    }
    int nhits = DefaultHistoShapeParametrizationNumberOfHits;
    // For debugging, set to producing nhits hits for each shape (by default
    // Poisson distribution around nhits)
    inhist->Scale( nhits / inhist->Integral() );

    TFCSHistoLateralShapeParametrization* shapeparam = nullptr;
    if ( cs < 21 ) {
      shapeparam = new TFCSHistoLateralShapeParametrization( shapepara_name, shapepara_title );
    } else {
      shapeparam = new TFCSHistoLateralShapeParametrizationFCal( shapepara_name, shapepara_title );
    }
    shapeparam->Initialize( inhist );
    shapeparam->set_pdgid( pdgId );
    if ( pdgId == 11 || pdgId == 211 ) shapeparam->add_pdgid( -pdgId );
    shapeparam->set_eta_min( etaMin );
    shapeparam->set_eta_max( etaMax );
    shapeparam->set_eta_nominal( 0.5 * ( etaMin + etaMax ) );
    shapeparam->set_Ekin_min( Ekin_min );
    shapeparam->set_Ekin_max( Ekin_max );
    shapeparam->set_Ekin_nominal( Ekin );
    shapeparam->set_calosample( cs );
    shapeparam->set_Ekin_bin( Ebin );
    shapeparam->set_phi_symmetric();
#ifdef FCS_DEBUG
    shapeparam->Print();
#endif
    shapefile->Close();

    return dynamic_cast<TFCSLateralShapeParametrizationHitBase*>( shapeparam );
  }

  TFCSParametrizationEbinChain* NewShapeEbinCaloSampleChain( TFCSParametrizationBase*                epara,
                                                             const LateralShapeParametrizationArray& mapping,
                                                             const LateralShapeParametrizationArray& numbersOfHits,
                                                             std::string shapeFileName, int pdgId, int intMom,
                                                             double etaMin, double etaMax ) {
    int int_etaMin = TMath::Nint( 100 * etaMin );
    int int_etaMax = TMath::Nint( 100 * etaMax );

    TString para_name  = Form( "Param_id%d_Mom%d_eta_%d_%d", pdgId, intMom, int_etaMin, int_etaMax );
    TString para_title = Form( "Param for id=%d Mom=%d %4.2f<eta<%4.2f", pdgId, intMom, etaMin, etaMax );
    TFCSParametrizationEbinChain* para = new TFCSParametrizationEbinChain( para_name, para_title );

    std::string extrapolFileName = shapeFileName;
    size_t      index            = extrapolFileName.rfind( ".shapepara." );
    if ( index != std::string::npos ) extrapolFileName.replace( index, 11, ".extrapol." );

    const int max_nbin = 100;

    for ( int iEbin = 0; iEbin <= max_nbin; ++iEbin ) {
      if ( !epara->is_match_Ekin_bin( iEbin ) ) continue;
      for ( int ilayer = 0; ilayer < 24; ++ilayer ) {
        if ( !epara->is_match_calosample( ilayer ) ) continue;

        TFCSLateralShapeParametrizationHitBase* centerPosCalc =
            NewCenterPositionCalculation( extrapolFileName, pdgId, intMom, etaMin, etaMax, iEbin, ilayer );

        TFCSLateralShapeParametrizationHitBase* shape =
            NewHistoShapeParametrization( shapeFileName, pdgId, intMom, etaMin, etaMax, iEbin, ilayer );
        if ( !shape ) return nullptr;

        TString chain_name =
            Form( "ShapeChain_id%d_Mom%d_eta_%d_%d_Ebin%d_cs%d", pdgId, intMom, int_etaMin, int_etaMax, iEbin, ilayer );
        TString chain_title = Form( "Shape chain for id=%d Mom=%d %4.2f<eta<%4.2f Ebin=%d cs=%d", pdgId, intMom, etaMin,
                                    etaMax, iEbin, ilayer );

        TFCSLateralShapeParametrizationHitChain* hitchain =
            new TFCSLateralShapeParametrizationHitChain( chain_name, chain_title );
        hitchain->set_pdgid_Ekin_eta_Ekin_bin_calosample( *shape );
        if ( centerPosCalc ) { hitchain->push_back( centerPosCalc ); }
        hitchain->push_back( shape );

        if ( !numbersOfHits.empty() ) { hitchain->set_number_of_hits_simul( numbersOfHits[ilayer] ); }

        hitchain->push_back( mapping[ilayer] );
        para->push_back_in_bin( hitchain, iEbin );
      }
    }

    return para;
  }

  TFCSParametrizationBase* NewEnergyChain( CLHEP::HepRandomEngine*                      randEngine,
                                           const FCS::LateralShapeParametrizationArray& mapping,
                                           const FCS::LateralShapeParametrizationArray& numbersOfHits, int pdgid,
                                           int int_Mom_min, int int_Mom_max, double etamin, double etamax ) {
    auto sample = std::make_unique<TFCSSampleDiscovery>();

    int     int_etamin = TMath::Nint( 100 * etamin );
    int     int_etamax = TMath::Nint( 100 * etamax );
    TString Ekinpara_name =
        Form( "SelEkin_id%d_Mom%d_%d_eta_%d_%d", pdgid, int_Mom_min, int_Mom_max, int_etamin, int_etamax );
    TString Ekinpara_title =
        Form( "Select Ekin for id=%d %d<=Mom<=%d %4.2f<=|eta|<%4.2f", pdgid, int_Mom_min, int_Mom_max, etamin, etamax );
    TFCSParametrizationEkinSelectChain* EkinSelectChain =
        new TFCSParametrizationEkinSelectChain( Ekinpara_name, Ekinpara_title );
    EkinSelectChain->set_DoRandomInterpolation();

    bool    addinit = false;
    TGraph* gr      = nullptr;
    // TODO: its a hack for now, should have a method in FCS_dsid.h to get this
    TString EinterfileName = sample->getEinterpolMeanName( pdgid );
    TString graph          = Form( "Graph_%i_%i", int_etamin, int_etamax );

    auto Einterfile = std::unique_ptr<TFile>( TFile::Open( EinterfileName ) );
    if ( !Einterfile ) {
      std::cerr << "================ interpolation ERROR ===============" << std::endl
                << "ERROR: pdgid =" << pdgid << " " << etamin << "<=eta<" << etamax << std::endl
                << "ERROR: Interpolation file=" << EinterfileName << std::endl
                << "ERROR: graph=" << graph << std::endl
                << "====================================================" << std::endl;
      std::cerr << "Error: Could not open file '" << EinterfileName << "'" << std::endl;
      gSystem->Exit( 1 );
    }

    std::cout << "================ interpolation ===============" << std::endl
              << "pdgid =" << pdgid << " " << etamin << "<=eta<" << etamax << std::endl
              << "Interpolation file=" << EinterfileName << std::endl
              << "graph=" << graph << std::endl
              << "===============================================" << std::endl;
    gr = (TGraph*)Einterfile->Get( graph );
    // gr = (TGraphErrors*)cv->GetPrimitive("Graph");
    Einterfile->Close();

    if ( !gr ) addinit = true;

    for ( int int_Mom = int_Mom_min; int_Mom <= int_Mom_max; int_Mom *= 2 ) {

      int                  dsid          = sample->findDSID( pdgid, int_Mom, int_etamin, 0 ).dsid;
      std::string          Eparafilename = sample->getSecondPCAName( dsid );
      std::string          shapefilename = sample->getShapeName( dsid );
      TFCSParametrization* para = NewParametrization( randEngine, mapping, numbersOfHits, Eparafilename, shapefilename,
                                                      pdgid, int_Mom, etamin, etamax, addinit );

      // Eparafilename="/afs/cern.ch/user/s/schaarsc/public/fastcalo/rel21/athena/Simulation/ISF/ISF_FastCaloSim/EnergyParametrization/scripts/output/ds430001.secondPCA.ver01.root";

      if ( para ) {
        if ( int_Mom == int_Mom_min ) para->set_Ekin_min( 0 );
        if ( int_Mom == int_Mom_max ) para->set_Ekin_max( 14000000 );
        EkinSelectChain->push_back_in_bin( para );
        std::cout << "========== EkinSelectChain ==========" << std::endl
                  << "pdgid=" << pdgid << " Mom=" << int_Mom << " " << etamin << "<=eta<" << etamax << std::endl
                  << "Efile=" << Eparafilename << std::endl
                  << "shapefile=" << shapefilename << std::endl
                  << "=====================================" << std::endl;
      } else {
        std::cout << "============= ERROR =================" << std::endl
                  << "ERROR: pdgid=" << pdgid << " Mom=" << int_Mom << " " << etamin << "<=eta<" << etamax << std::endl
                  << "ERROR: Efile=" << Eparafilename << std::endl
                  << "ERROR: shapefile=" << shapefilename << std::endl
                  << "=====================================" << std::endl;
        gSystem->Exit( 1 );
      }
    }

    if ( gr ) {
      TString Einterpara_name =
          Form( "SplineInterpolEkin_id%d_Mom%d_%d_eta_%d_%d", pdgid, int_Mom_min, int_Mom_max, int_etamin, int_etamax );
      TString Einterpara_title = Form( "Spline interpolation Ekin for id=%d %d<=Mom<=%d %4.2f<=|eta|<%4.2f", pdgid,
                                       int_Mom_min, int_Mom_max, etamin, etamax );
      TFCSEnergyInterpolationSpline* Einterpol = new TFCSEnergyInterpolationSpline( Einterpara_name, Einterpara_title );
      Einterpol->set_pdgid_Ekin_eta( *EkinSelectChain );
      Einterpol->InitFromArrayInEkin( gr->GetN(), gr->GetX(), gr->GetY(), "b2e2", 0, 0 );
      EkinSelectChain->push_before_first_bin( Einterpol );
    }

    return (TFCSParametrizationBase*)EkinSelectChain;
  }

  TFCSEnergyParametrization* NewPCAEnergyParametrization( CLHEP::HepRandomEngine* randEngine, std::string filename,
                                                          int pdgid, int int_Mom, double etamin, double etamax ) {
    double Ekin     = TFCSAnalyzerBase::Mom2Ekin( pdgid, int_Mom );
    double Ekin_min = TFCSAnalyzerBase::Mom2Ekin_min( pdgid, int_Mom );
    double Ekin_max = TFCSAnalyzerBase::Mom2Ekin_max( pdgid, int_Mom );

    int     int_etamin  = TMath::Nint( 100 * etamin );
    int     int_etamax  = TMath::Nint( 100 * etamax );
    TString epara_name  = Form( "Epara_id%d_Mom%d_eta_%d_%d", pdgid, int_Mom, int_etamin, int_etamax );
    TString epara_title = Form( "Energy param for id=%d Mom=%d %4.2f<eta<%4.2f", pdgid, int_Mom, etamin, etamax );

    TFCSPCAEnergyParametrization* epara = new TFCSPCAEnergyParametrization( epara_name, epara_title );
    epara->set_pdgid( pdgid );
    if ( pdgid == 11 || pdgid == 211 ) epara->add_pdgid( -pdgid );
    epara->set_eta_min( etamin );
    epara->set_eta_max( etamax );
    epara->set_eta_nominal( 0.5 * ( etamin + etamax ) );
    epara->set_Ekin_min( Ekin_min );
    epara->set_Ekin_max( Ekin_max );
    epara->set_Ekin_nominal( Ekin );
    auto eparafile = std::unique_ptr<TFile>( TFile::Open( filename.c_str() ) );
    if ( !eparafile ) {
      std::cerr << "Error: Could not open file '" << filename << "'" << std::endl;
      return nullptr;
    }

    if ( !epara->loadInputs( eparafile.get(), "" ) ) return nullptr;
    eparafile->Close();

    // if(epara->n_bins()==1) return nullptr;
    if ( epara->get_layers().size() == 0 ) return nullptr;

    TFCSSimulationState simulstate( randEngine );
    MeanAndRMS          meanEresponse;
    int                 ntoy = 0;
    for ( int itoy = 0; itoy < 100000; ++itoy ) {
      for ( int ibin = 1; ibin <= epara->n_bins(); ++ibin ) {
        simulstate.set_E( Ekin );
        simulstate.set_Ebin( ibin );
        epara->simulate( simulstate, nullptr, nullptr );
        meanEresponse.add( simulstate.E() );
        ++ntoy;
      }
      if ( itoy > 100 && meanEresponse.mean_error() / meanEresponse.mean() < 0.001 ) break;
    }

    std::cout << "Emean=" << meanEresponse.mean() << " +- " << meanEresponse.mean_error() << " (" << ntoy << " toys)"
              << std::endl;
    epara->set_Ekin_nominal( meanEresponse.mean() );

    // #ifdef FCS_DEBUG
    // epara->Print();
    // #endif
    return (TFCSEnergyParametrization*)epara;
  }

  TFCSParametrization* NewParametrization( CLHEP::HepRandomEngine*                      randEngine,
                                           const FCS::LateralShapeParametrizationArray& mapping,
                                           const FCS::LateralShapeParametrizationArray& numbersOfHits,
                                           std::string Eparafilename, std::string shapefilename, int pdgid, int int_Mom,
                                           double etamin, double etamax, bool addinit ) {
    double Ekin = TFCSAnalyzerBase::Mom2Ekin( pdgid, int_Mom );
    // double Ekin_min = (Ekin / 2) * TMath::Sqrt(2);
    // double Ekin_max = Ekin * TMath::Sqrt(2);
    int int_etamin = TMath::Nint( 100 * etamin );
    int int_etamax = TMath::Nint( 100 * etamax );

    TFCSEnergyParametrization* epara =
        NewPCAEnergyParametrization( randEngine, Eparafilename, pdgid, int_Mom, etamin, etamax );
    if ( !epara ) return nullptr;

    TFCSParametrizationEbinChain* para = FCS::NewShapeEbinCaloSampleChain( epara, mapping, numbersOfHits, shapefilename,
                                                                           pdgid, int_Mom, etamin, etamax );
    if ( !para ) return nullptr;

    /// create equal probability vector for other hadrons
    std::vector<float> prob( epara->n_bins() + 1, 0 );
    for ( int iEbin = 1; iEbin <= epara->n_bins(); ++iEbin ) prob[iEbin] = 1;

    auto eparafile = std::unique_ptr<TFile>( TFile::Open( Eparafilename.c_str() ) );
    if ( !eparafile ) {
      std::cerr << "================ PCA probability ERROR ===============" << std::endl
                << "ERROR: pdgid =" << pdgid << " " << etamin << "<=eta<" << etamax << std::endl
                << "ERROR: pca probability file=" << Eparafilename << std::endl
                << "====================================================" << std::endl;
      std::cerr << "Error: Could not open file '" << Eparafilename << "'" << std::endl;
      gSystem->Exit( 1 );
    }

    std::cout << "================ PCA probability ===============" << std::endl
              << "pdgid =" << pdgid << " " << etamin << "<=eta<" << etamax << std::endl
              << "pca prob  file =" << Eparafilename << std::endl
              << "===============================================" << std::endl;

    TString ebinpara_name  = Form( "ParamEbin_id%d_Mom%d_eta_%d_%d", pdgid, int_Mom, int_etamin, int_etamax );
    TString ebinpara_title = Form( "Param Ebin for id=%d Mom=%d %4.2f<eta<%4.2f", pdgid, int_Mom, etamin, etamax );
    TFCSEnergyBinParametrization* ebinpara = new TFCSEnergyBinParametrization( ebinpara_name, ebinpara_title );
    ebinpara->set_number_of_Ekin_bins( epara->n_bins() );
    ebinpara->set_pdgid_Ekin_eta( *epara );
    ebinpara->set_Ekin_nominal( Ekin );
    ebinpara->load_pdgid_Ekin_bin_probability_from_file( pdgid, eparafile.get(), "PCAbinprob" );
    if ( pdgid == 11 ) { ebinpara->load_pdgid_Ekin_bin_probability_from_file( -11, eparafile.get(), "PCAbinprob" ); }
    if ( pdgid == 211 ) {
      ebinpara->load_pdgid_Ekin_bin_probability_from_file( -211, eparafile.get(), "PCAbinprob" );
      /// to catch all other hadrons: use equal probability for now
      ebinpara->set_pdgid_Ekin_bin_probability( 0, prob );
      // ebinpara->load_pdgid_Ekin_bin_probability_from_file(0, eparafile,
      // "PCAbinprob");
    }

    if ( addinit ) {

      TString Einitpara_name =
          Form( "LinearInterpolEkin_id%d_Mom%d_eta_%d_%d", pdgid, int_Mom, int_etamin, int_etamax );
      TString Einitpara_title =
          Form( "Linear interpolation Ekin for id=%d Mom=%d %4.2f<eta<%4.2f", pdgid, int_Mom, etamin, etamax );
      TFCSEnergyInterpolationLinear* Einit = new TFCSEnergyInterpolationLinear( Einitpara_name, Einitpara_title );
      Einit->set_pdgid_Ekin_eta( *para );
      Einit->set_slope( epara->Ekin_nominal() / para->Ekin_nominal() );
      para->push_before_first_bin( Einit );
    }
    para->push_before_first_bin( ebinpara );
    para->push_before_first_bin( epara );

    return (TFCSParametrization*)para;
  }

  TFCSParametrizationBase* NewParametrizationSimple( CLHEP::HepRandomEngine*                      randEngine,
                                                     const FCS::LateralShapeParametrizationArray& mapping,
                                                     const FCS::LateralShapeParametrizationArray& numbersOfHits,
                                                     std::string Eparafilename, std::string shapefilename, int pdgid,
                                                     int int_Mom, double etamin, double etamax ) {
    int int_etamin = TMath::Nint( 100 * etamin );
    int int_etamax = TMath::Nint( 100 * etamax );

    TString para_name              = Form( "Param_id%d_Mom%d_eta_%d_%d", pdgid, int_Mom, int_etamin, int_etamax );
    TString para_title             = Form( "Param for id=%d Mom=%d %4.2f<eta<%4.2f", pdgid, int_Mom, etamin, etamax );
    TFCSParametrizationChain* para = new TFCSParametrizationChain( para_name, para_title );

    TFCSEnergyParametrization* epara =
        FCS::NewPCAEnergyParametrization( randEngine, Eparafilename, pdgid, int_Mom, etamin, etamax );
    para->push_back( epara );
    for ( int iEbin = 0; iEbin <= epara->n_bins(); ++iEbin ) {
      if ( !epara->is_match_Ekin_bin( iEbin ) ) continue;
      for ( int ilayer = 0; ilayer < 24; ++ilayer ) {
        if ( !epara->is_match_calosample( ilayer ) ) continue;
        TFCSLateralShapeParametrizationHitBase* shape =
            NewHistoShapeParametrization( shapefilename, pdgid, int_Mom, etamin, etamax, iEbin, ilayer );
        TString chain_name  = Form( "ShapeChain_id%d_Mom%d_eta_%d_%d_Ebin%d_cs%d", pdgid, int_Mom, int_etamin,
                                   int_etamax, iEbin, ilayer );
        TString chain_title = Form( "Shape chain for id=%d Mom=%d %4.2f<eta<%4.2f Ebin=%d cs=%d", pdgid, int_Mom,
                                    etamin, etamax, iEbin, ilayer );
        TFCSLateralShapeParametrizationHitChain* hitchain = new TFCSLateralShapeParametrizationHitChain( shape );
        hitchain->SetNameTitle( chain_name, chain_title );
        hitchain->set_number_of_hits_simul( numbersOfHits[ilayer] );
        hitchain->push_back( mapping[ilayer] );
        para->push_back( hitchain );
      }
    }

    return (TFCSParametrizationBase*)para;
  }

} // namespace FCS
