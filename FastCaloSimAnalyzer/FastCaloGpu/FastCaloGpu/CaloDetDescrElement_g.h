/*
  Copyright (C) 2002-2017 CERN for the benefit of the ATLAS collaboration
*/

#ifndef CALODETDESCRELEMENT_G_H
#define CALODETDESCRELEMENT_G_H

#ifndef ISF_FASTCALOSIMPARAMETRIZATION_CALODETDESCRELEMENT_H
#define ISF_FASTCALOSIMPARAMETRIZATION_CALODETDESCRELEMENT_H


#include "Identifier_g.h"

class CaloDetDescrElement {
 public:
__HOSTDEV__  CaloDetDescrElement() {
    m_identify = 0;
    m_hash_id = 0;
    m_calosample = 0;
    m_eta = 0;
    m_phi = 0;
    m_deta = 0;
    m_dphi = 0;
    m_r = 0;
    m_eta_raw = 0;
    m_phi_raw = 0;
    m_r_raw = 0;
    m_dr = 0;
    m_x = 0;
    m_y = 0;
    m_z = 0;
    m_x_raw = 0;
    m_y_raw = 0;
    m_z_raw = 0;
    m_dx = 0;
    m_dy = 0;
    m_dz = 0;
  };

  /** @brief virtual destructor
   */
__HOSTDEV__    virtual ~CaloDetDescrElement() {};

  /** @brief cell eta
   */
__HOSTDEV__    float eta() const;
  /** @brief cell phi
   */
__HOSTDEV__    float phi() const;
  /** @brief cell r
   */
__HOSTDEV__    float r() const;
  /** @brief cell eta_raw
   */
__HOSTDEV__    float eta_raw() const;
  /** @brief cell phi_raw
   */
__HOSTDEV__    float phi_raw() const;
  /** @brief cell r_raw
   */
__HOSTDEV__    float r_raw() const;
  /** @brief cell dphi
   */
__HOSTDEV__    float dphi() const;
  /** @brief cell deta
   */
__HOSTDEV__    float deta() const;
  /** @brief cell dr
   */
__HOSTDEV__    float dr() const;

  /** @brief cell x
   */
__HOSTDEV__    float x() const;
  /** @brief cell y
   */
__HOSTDEV__    float y() const;
  /** @brief cell z
   */
__HOSTDEV__    float z() const;
  /** @brief cell x_raw
   */
__HOSTDEV__    float x_raw() const;
  /** @brief cell y_raw
   */
__HOSTDEV__    float y_raw() const;
  /** @brief cell z_raw
   */
__HOSTDEV__    float z_raw() const;
  /** @brief cell dx
   */
__HOSTDEV__    float dx() const;
  /** @brief cell dy
   */
__HOSTDEV__    float dy() const;
  /** @brief cell dz
   */
__HOSTDEV__    float dz() const;

  /** @brief cell identifier
   */
__HOSTDEV__    Identifier identify() const;

__HOSTDEV__    unsigned long long calo_hash() const;

__HOSTDEV__    int getSampling() const ;

 //ACH protected:
 //
  long long m_identify;
  long long m_hash_id;

  int m_calosample;

  /** @brief cylindric coordinates : eta
   */
  float m_eta;
  /** @brief cylindric coordinates : phi
   */
  float m_phi;

 /** @brief this one is cached for algorithm working in transverse Energy
    */
  float m_sinTh;
   /** @brief this one is cached for algorithm working in transverse Energy
   */
  float m_cosTh;

  /** @brief cylindric coordinates : delta eta
   */
  float m_deta;
  /** @brief cylindric coordinates : delta phi
   */
  float m_dphi;

  /** @brief cylindric coordinates : r
   */

  float m_volume;

  /** @brief cache to allow fast px py pz computation
   */
  float m_sinPhi;

  /** @brief cache to allow fast px py pz computation
  */
  float m_cosPhi;

  /** @brief cylindric coordinates : r
   */
  float m_r;
  /** @brief cylindric coordinates : eta_raw
   */
  float m_eta_raw;
  /** @brief cylindric coordinates : phi_raw
   */
  float m_phi_raw;
  /** @brief cylindric coordinates : r_raw
   */
  float m_r_raw;
  /** @brief cylindric coordinates : delta r
   */
  float m_dr;

  /** @brief cartesian coordinates : X
   */
  float m_x;
  /** @brief cartesian coordinates : Y
   */
  float m_y;
  /** @brief cartesian coordinates : Z
   */
  float m_z;
  /** @brief cartesian coordinates : X raw
   */
  float m_x_raw;
  /** @brief cartesian coordinates : Y raw
   */
  float m_y_raw;
  /** @brief cartesian coordinates : Z raw
   */
  float m_z_raw;
  /** @brief cartesian coordinates : delta X
   */
  float m_dx;
  /** @brief cartesian coordinates : delta Y
   */
  float m_dy;
  /** @brief cartesian coordinates : delta Z
   */
  float m_dz;

};

__HOSTDEV__ __INLINE__ Identifier CaloDetDescrElement::identify() const {
	Identifier id((unsigned long long) m_identify);
	return id;
}

__HOSTDEV__ __INLINE__ unsigned long long CaloDetDescrElement::calo_hash() const { return m_hash_id; }

__HOSTDEV__ __INLINE__ int   CaloDetDescrElement::getSampling() const { return m_calosample; }
__HOSTDEV__ __INLINE__ float CaloDetDescrElement::eta() const { return m_eta; }
__HOSTDEV__ __INLINE__ float CaloDetDescrElement::phi() const { return m_phi; }
__HOSTDEV__ __INLINE__ float CaloDetDescrElement::r() const { return m_r; }
__HOSTDEV__ __INLINE__ float CaloDetDescrElement::eta_raw() const { return m_eta_raw; }
__HOSTDEV__ __INLINE__ float CaloDetDescrElement::phi_raw() const { return m_phi_raw; }
__HOSTDEV__ __INLINE__ float CaloDetDescrElement::r_raw() const { return m_r_raw; }
__HOSTDEV__ __INLINE__ float CaloDetDescrElement::deta() const { return m_deta; }
__HOSTDEV__ __INLINE__ float CaloDetDescrElement::dphi() const { return m_dphi; }
__HOSTDEV__ __INLINE__ float CaloDetDescrElement::dr() const { return m_dr; }

__HOSTDEV__ __INLINE__ float CaloDetDescrElement::x() const { return m_x; }
__HOSTDEV__ __INLINE__ float CaloDetDescrElement::y() const { return m_y; }
__HOSTDEV__ __INLINE__ float CaloDetDescrElement::z() const { return m_z; }
__HOSTDEV__ __INLINE__ float CaloDetDescrElement::x_raw() const { return m_x_raw; }
__HOSTDEV__ __INLINE__ float CaloDetDescrElement::y_raw() const { return m_y_raw; }
__HOSTDEV__ __INLINE__ float CaloDetDescrElement::z_raw() const { return m_z_raw; }
__HOSTDEV__ __INLINE__ float CaloDetDescrElement::dx() const { return m_dx; }
__HOSTDEV__ __INLINE__ float CaloDetDescrElement::dy() const { return m_dy; }
__HOSTDEV__ __INLINE__ float CaloDetDescrElement::dz() const { return m_dz; }

#endif
#endif
