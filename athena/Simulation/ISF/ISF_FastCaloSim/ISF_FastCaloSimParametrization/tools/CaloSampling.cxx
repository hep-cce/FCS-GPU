/*
  Copyright (C) 2002-2017 CERN for the benefit of the ATLAS collaboration
*/

#include "CaloGeoHelpers/CaloSampling.h"

namespace {

const char* const sample_names[] = {
#define CALOSAMPLING(name, inbarrel, inendcap) #name ,
#include "CaloGeoHelpers/CaloSampling.def"
#undef CALOSAMPLING
};

} // anonymous namespace

unsigned int CaloSampling::getNumberOfSamplings()
{
  return (unsigned int)Unknown;
}


std::string CaloSampling::getSamplingName (CaloSample theSample)
{
  return sample_names[theSample];
}


std::string CaloSampling::getSamplingName (unsigned int theSample)
{
  if (theSample >= getNumberOfSamplings())
    return "";
  return sample_names[theSample];
}
