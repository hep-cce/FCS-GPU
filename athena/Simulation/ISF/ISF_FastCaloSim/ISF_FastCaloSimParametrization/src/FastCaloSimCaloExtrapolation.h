/*
  Copyright (C) 2002-2017 CERN for the benefit of the ATLAS collaboration
*/

#ifndef FastCaloSimCaloExtrapolation_H
#define FastCaloSimCaloExtrapolation_H

// Athena includes
#include "AthenaBaseComps/AthAlgTool.h"
#include "GaudiKernel/ToolHandle.h"

#include <vector>

namespace Trk
{
  class TrackingVolume;
}

#include "TrkExInterfaces/ITimedExtrapolator.h"
#include "TrkEventPrimitives/PdgToParticleHypothesis.h"

class ICaloSurfaceHelper;

namespace HepPDT
{
  class ParticleDataTable;
}

#include "ISF_FastCaloSimParametrization/IFastCaloSimCaloExtrapolation.h"
#include "ISF_FastCaloSimEvent/FastCaloSim_CaloCell_ID.h"
#include "ISF_FastCaloSimEvent/TFCSExtrapolationState.h"

#include "ISF_FastCaloSimParametrization/IFastCaloSimGeometryHelper.h"


class FastCaloSimCaloExtrapolation:public AthAlgTool, virtual public IFastCaloSimCaloExtrapolation
{

public:
  FastCaloSimCaloExtrapolation( const std::string& t, const std::string& n, const IInterface* p );
  ~FastCaloSimCaloExtrapolation();

  virtual StatusCode initialize() override final;
  virtual StatusCode finalize() override final;

  enum SUBPOS { SUBPOS_MID = TFCSExtrapolationState::SUBPOS_MID, SUBPOS_ENT = TFCSExtrapolationState::SUBPOS_ENT, SUBPOS_EXT = TFCSExtrapolationState::SUBPOS_EXT}; //MID=middle, ENT=entrance, EXT=exit of cal layer

  virtual void extrapolate(TFCSExtrapolationState& result,const TFCSTruthState* truth) override final;

protected:
  const IFastCaloSimGeometryHelper* GetCaloGeometry() const {return &(*m_CaloGeometryHelper);};

  // extrapolation through Calo
  std::vector<Trk::HitInfo>* caloHits(const TFCSTruthState* truth) const;
  void extrapolate(TFCSExtrapolationState& result,const TFCSTruthState* truth,std::vector<Trk::HitInfo>* hitVector);
  void extrapolate_to_ID(TFCSExtrapolationState& result,const TFCSTruthState* truth,std::vector<Trk::HitInfo>* hitVector);
  bool get_calo_etaphi(TFCSExtrapolationState& result,std::vector<Trk::HitInfo>* hitVector,int sample,int subpos=SUBPOS_MID);
  bool get_calo_surface(TFCSExtrapolationState& result,std::vector<Trk::HitInfo>* hitVector);
  bool rz_cylinder_get_calo_etaphi(std::vector<Trk::HitInfo>* hitVector, double cylR, double cylZ, Amg::Vector3D& pos, Amg::Vector3D& mom);

  bool   isCaloBarrel(int sample) const;
  double deta(int sample,double eta) const;
  void   minmaxeta(int sample,double eta,double& mineta,double& maxeta) const;
  double rzmid(int sample,double eta) const;
  double rzent(int sample,double eta) const;
  double rzext(int sample,double eta) const;
  double rmid(int sample,double eta) const;
  double rent(int sample,double eta) const;
  double rext(int sample,double eta) const;
  double zmid(int sample,double eta) const;
  double zent(int sample,double eta) const;
  double zext(int sample,double eta) const;
  double rpos(int sample,double eta,int subpos = CaloSubPos::SUBPOS_MID) const;
  double zpos(int sample,double eta,int subpos = CaloSubPos::SUBPOS_MID) const;
  double rzpos(int sample,double eta,int subpos = CaloSubPos::SUBPOS_MID) const;

  HepPDT::ParticleDataTable*     m_particleDataTable{nullptr};

  //Define ID-CALO surface to be used for AFII 
  //TODO: this should eventually extrapolate to a uniquly defined surface!
  std::vector<double> m_CaloBoundaryR{1148.0,120.0,41.0};
  std::vector<double> m_CaloBoundaryZ{3550.0,4587.0,4587.0};
  double m_calomargin{100};

  std::vector< int > m_surfacelist;

  // The new Extrapolator setup
  ToolHandle<Trk::ITimedExtrapolator> m_extrapolator;
  ToolHandle<ICaloSurfaceHelper>      m_caloSurfaceHelper;
  mutable const Trk::TrackingVolume*  m_caloEntrance{nullptr};
  std::string                         m_caloEntranceName{""};

  Trk::PdgToParticleHypothesis        m_pdgToParticleHypothesis;

  // The FastCaloSimGeometryHelper tool
  ToolHandle<IFastCaloSimGeometryHelper> m_CaloGeometryHelper;
};

#endif // FastCaloSimCaloExtrapolation_H
