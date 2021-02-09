/*
  Copyright (C) 2002-2021 CERN for the benefit of the ATLAS collaboration
*/

#ifndef ParticleID_h
#define ParticleID_h



namespace HepPDT{
   
   class ParticleID{
     public: 
      ParticleID(const int pdgID);
      inline double charge(){return m_charge;}
     private:
      double m_charge;
   };
   
   
}




#if defined(__ROOTCLING__) && defined(__FastCaloSimStandAlone__)
#pragma link C++ class ParticleID+;
#endif

#endif
