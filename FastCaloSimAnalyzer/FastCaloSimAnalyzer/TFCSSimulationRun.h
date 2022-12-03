/*
  Copyright (C) 2002-2023 CERN for the benefit of the ATLAS collaboration
*/

#ifndef TFCSSimulationRun_H
#define TFCSSimulationRun_H

#include "ISF_FastCaloSimEvent/TFCSParametrizationBase.h"
#include "ISF_FastCaloSimEvent/TFCSSimulationState.h"

class TFCSSimulationRun: public TNamed
{
public:
  TFCSSimulationRun(const char* name=0, const char* title=0):TNamed(name,title),m_basesim(0),m_simul(0) {};
  TFCSSimulationRun(const char* name, const char* title,TFCSParametrizationBase* ref):TNamed(name,title),m_basesim(ref),m_simul(0) {};
  TFCSSimulationRun(TFCSParametrizationBase* ref):TNamed(*ref),m_basesim(ref),m_simul(0) {};
  TFCSSimulationRun(TFCSParametrizationBase* ref, std::vector<TFCSSimulationState>& refsimul):TNamed(*ref),m_basesim(ref),m_simul(refsimul) {};

  TFCSParametrizationBase* basesim() {return m_basesim;};
  std::vector<TFCSSimulationState>& simul() {return m_simul;};
private:
  TFCSParametrizationBase* m_basesim;
  std::vector<TFCSSimulationState> m_simul;

  ClassDef(TFCSSimulationRun, 1);
};

#if defined(__MAKECINT__)
#pragma link C++ class TFCSSimulationRun+;
#endif

#endif
