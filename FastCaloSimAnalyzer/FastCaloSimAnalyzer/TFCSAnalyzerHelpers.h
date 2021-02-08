/*
  Copyright (C) 2002-2018 CERN for the benefit of the ATLAS collaboration
*/

#include <vector>

#define DefaultHistoShapeParametrizationNumberOfHits 50000

namespace CLHEP {
  class HepRandomEngine;
}

class TFCSEnergyParametrization;
class TFCSLateralShapeParametrizationHitBase;
class TFCSParametrization;
class TFCSParametrizationBase;
class TFCSParametrizationEbinChain;

namespace FCS {

  typedef std::array<TFCSLateralShapeParametrizationHitBase*, 24> LateralShapeParametrizationArray;

  void init_hit_to_cell_mapping( LateralShapeParametrizationArray& mapping, bool isNewWiggle = true );
  void init_hit_to_cell_mapping_with_wiggle( LateralShapeParametrizationArray& mapping, int sampling,
                                             const std::vector<std::string>& etaRange,
                                             const std::vector<float>&       etaLowEdge,
                                             const std::vector<float>&       cellDphiHalve = {} );

  void init_hit_to_cell_mapping_with_wiggle( LateralShapeParametrizationArray& mapping, int sampling, double rangeMin,
                                             double rangeMax, const std::vector<float>& cellDphiHalve = {} );

  void init_numbers_of_hits( LateralShapeParametrizationArray& mapping, float scale = 1 );

  TFCSLateralShapeParametrizationHitBase* NewCenterPositionCalculation( std::string fileName, int pdgId, int intMom,
                                                                        double etaMin, double etaMax, int Ebin,
                                                                        int cs );

  TFCSLateralShapeParametrizationHitBase* NewHistoShapeParametrization( std::string fileName, int pdgId, int intMom,
                                                                        double etaMin, double etaMax, int Ebin,
                                                                        int cs );

  TFCSParametrizationEbinChain* NewShapeEbinCaloSampleChain( TFCSParametrizationBase*                epara,
                                                             const LateralShapeParametrizationArray& mapping,
                                                             const LateralShapeParametrizationArray& numbersOfHits,
                                                             std::string shapeFileName, int pdgId, int intMom,
                                                             double etaMin, double etaMax );

  TFCSParametrizationBase* NewEnergyChain( CLHEP::HepRandomEngine*                      randEngine,
                                           const FCS::LateralShapeParametrizationArray& mapping,
                                           const FCS::LateralShapeParametrizationArray& numbersOfHits, int pdgid,
                                           int int_Mom_min, int int_Mom_max, double etamin, double etamax );

  TFCSEnergyParametrization* NewPCAEnergyParametrization( CLHEP::HepRandomEngine* randEngine, std::string filename,
                                                          int pdgid, int int_Mom, double etamin, double etamax );

  TFCSParametrization* NewParametrization( CLHEP::HepRandomEngine*                      randEngine,
                                           const FCS::LateralShapeParametrizationArray& mapping,
                                           const FCS::LateralShapeParametrizationArray& numbersOfHits,
                                           std::string Eparafilename, std::string shapefilename, int pdgid, int int_Mom,
                                           double etamin, double etamax, bool addinit = false );

  TFCSParametrizationBase* NewParametrizationSimple( CLHEP::HepRandomEngine*                      randEngine,
                                                     const FCS::LateralShapeParametrizationArray& mapping,
                                                     const FCS::LateralShapeParametrizationArray& numbersOfHits,
                                                     std::string Eparafilename, std::string shapefilename, int pdgid,
                                                     int int_Mom, double etamin, double etamax );

} // namespace FCS
