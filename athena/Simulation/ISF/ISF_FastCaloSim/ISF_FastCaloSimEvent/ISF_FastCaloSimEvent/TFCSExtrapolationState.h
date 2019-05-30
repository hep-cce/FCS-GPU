/*
  Copyright (C) 2002-2018 CERN for the benefit of the ATLAS collaboration
*/

#ifndef ISF_FASTCALOSIMEVENT_TFCSExtrapolationState_h
#define ISF_FASTCALOSIMEVENT_TFCSExtrapolationState_h

#include <TObject.h>
#include "ISF_FastCaloSimEvent/FastCaloSim_CaloCell_ID.h"

class TFCSExtrapolationState:public TObject {
  public:
    TFCSExtrapolationState();

    void clear();

    enum SUBPOS { SUBPOS_MID = 0, SUBPOS_ENT = 1, SUBPOS_EXT = 2}; //MID=middle, ENT=entrance, EXT=exit of cal layer

    void set_OK (int layer,int subpos,bool   val=true) {m_CaloOK[layer][subpos]=val;};

    void set_eta(int layer,int subpos,double val) {m_etaCalo[layer][subpos]=val;};
    void set_phi(int layer,int subpos,double val) {m_phiCalo[layer][subpos]=val;};
    void set_r  (int layer,int subpos,double val) {m_rCalo[layer][subpos]=val;};
    void set_z  (int layer,int subpos,double val) {m_zCalo[layer][subpos]=val;};
    void set_d  (int layer,int subpos,double val) {m_dCalo[layer][subpos]=val;};
    void set_detaBorder(int layer,int subpos,double val) {m_distetaCaloBorder[layer][subpos]=val;};

    void set_CaloSurface_sample(int val) {m_CaloSurface_sample=val;};
    void set_CaloSurface_eta(double val) {m_CaloSurface_eta=val;};
    void set_CaloSurface_phi(double val) {m_CaloSurface_phi=val;};
    void set_CaloSurface_r(double val) {m_CaloSurface_r=val;};
    void set_CaloSurface_z(double val) {m_CaloSurface_z=val;};

    void set_IDCaloBoundary_eta(double val) {m_IDCaloBoundary_eta=val;};
    void set_IDCaloBoundary_phi(double val) {m_IDCaloBoundary_phi=val;};
    void set_IDCaloBoundary_r(double val) {m_IDCaloBoundary_r=val;};
    void set_IDCaloBoundary_z(double val) {m_IDCaloBoundary_z=val;};

    bool   OK(int layer,int subpos) const {return m_CaloOK[layer][subpos];};
    double eta(int layer,int subpos) const {return m_etaCalo[layer][subpos];};
    double phi(int layer,int subpos) const {return m_phiCalo[layer][subpos];};
    double r(int layer,int subpos) const {return m_rCalo[layer][subpos];};
    double z(int layer,int subpos) const {return m_zCalo[layer][subpos];};
    double d(int layer,int subpos) const {return m_dCalo[layer][subpos];};
    double detaBorder(int layer,int subpos) const {return m_distetaCaloBorder[layer][subpos];};

    int    CaloSurface_sample() const {return m_CaloSurface_sample;};
    double CaloSurface_eta() const {return m_CaloSurface_eta;};
    double CaloSurface_phi() const {return m_CaloSurface_phi;};
    double CaloSurface_r() const {return m_CaloSurface_r;};
    double CaloSurface_z() const {return m_CaloSurface_z;};

    double IDCaloBoundary_eta() const {return m_IDCaloBoundary_eta;};
    double IDCaloBoundary_phi() const {return m_IDCaloBoundary_phi;};
    double IDCaloBoundary_r() const {return m_IDCaloBoundary_r;};
    double IDCaloBoundary_z() const {return m_IDCaloBoundary_z;};

    double IDCaloBoundary_AngleEta() const {return m_IDCaloBoundary_AngleEta;};
    double IDCaloBoundary_Angle3D() const {return m_IDCaloBoundary_Angle3D;};

    void set_IDCaloBoundary_AngleEta(double val) {m_IDCaloBoundary_AngleEta=val;};
    void set_IDCaloBoundary_Angle3D(double val) {m_IDCaloBoundary_Angle3D=val;};
	
	void Print(Option_t *option="") const;

  private:
    bool   m_CaloOK[CaloCell_ID_FCS::MaxSample][3];
    double m_etaCalo[CaloCell_ID_FCS::MaxSample][3];
    double m_phiCalo[CaloCell_ID_FCS::MaxSample][3];
    double m_rCalo[CaloCell_ID_FCS::MaxSample][3];
    double m_zCalo[CaloCell_ID_FCS::MaxSample][3];
    double m_dCalo[CaloCell_ID_FCS::MaxSample][3];
    double m_distetaCaloBorder[CaloCell_ID_FCS::MaxSample][3];

    int    m_CaloSurface_sample;
    double m_CaloSurface_eta;
    double m_CaloSurface_phi;
    double m_CaloSurface_r;
    double m_CaloSurface_z;

    double m_IDCaloBoundary_eta;
    double m_IDCaloBoundary_phi;
    double m_IDCaloBoundary_r;
    double m_IDCaloBoundary_z;

    double m_IDCaloBoundary_AngleEta;
    double m_IDCaloBoundary_Angle3D;

  ClassDef(TFCSExtrapolationState,1)  //TFCSExtrapolationState
};

#if defined(__ROOTCLING__) && defined(__FastCaloSimStandAlone__)
#pragma link C++ class TFCSExtrapolationState+;
#endif

#endif
