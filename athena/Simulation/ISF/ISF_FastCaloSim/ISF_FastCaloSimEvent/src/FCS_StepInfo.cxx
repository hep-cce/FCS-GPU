/*
  Copyright (C) 2002-2017 CERN for the benefit of the ATLAS collaboration
*/

#include "ISF_FastCaloSimEvent/FCS_StepInfo.h"

#include "GaudiKernel/MsgStream.h"


/*
ISF_FCS_Parametrization::FCS_StepInfo::FCS_StepInfo(const FCS_StepInfo& first, const FCS_StepInfo& second)
{
  double esum = first.m_energy + second.m_energy;
  double w1 = 0;
  double w2 = 0;

  if (esum > 0) {
    w1 =  first.m_energy/esum;
    w2 =  second.m_energy/esum;
  }

  m_pos = w1*first.m_pos + w2*second.m_pos;
  m_time = w1*first.m_time + w2*second.m_time;
  m_energy = esum;
  m_valid = true;
  m_detector = first.m_detector;  //need to make sure that it's not merging hits from different detector parts..
  m_ID = first.m_ID; //dtto
}
*/
double ISF_FCS_Parametrization::FCS_StepInfo::diff2(const FCS_StepInfo& other) const
  {
    return (this->position().diff2(other.position()));
  }


ISF_FCS_Parametrization::FCS_StepInfo& ISF_FCS_Parametrization::FCS_StepInfo::operator+=(const ISF_FCS_Parametrization::FCS_StepInfo& other)
{
  if (identify() != other.identify())
    {
      std::cout <<"Warning: Not merging hits from different cells!!! "<<identify()<<" / "<<other.identify()<<std::endl;
      return *this;
    }

  if ( (fabs( energy() ) > 1e-9) && (fabs( other.energy() ) > 1e-9))
    {
      //both !=0
      //Use absolute energies for weighting
      double eabssum = fabs(energy())+fabs(other.energy());
      double esum = energy()+other.energy();
      double w1 =  fabs(energy())/eabssum;
      double w2 =  fabs(other.energy())/eabssum;
      //Average position, time, energy sum
      m_pos = w1*m_pos + w2*other.m_pos;
      setEnergy(esum);
      setTime(w1* time()+ w2 * other.time());


    }
  else if (fabs( energy() ) < 1e-9)
      {
        //original is 0, use other
        setEnergy(other.energy());
        setP(other.position());
        setTime(other.time());
      }
  else if (fabs( other.energy() ) < 1e-9)
    {
      //other is 0, use original
      //don't need to do anything...
    }
  else
    {
      std::cout <<"Warning: merging hits something weird: "<<std::endl;
      std::cout <<"Original hit: "<<energy()<<" "<<position()<<std::endl;
      std::cout <<"Second hit: "<<other.energy()<<" "<<other.position()<<std::endl;
    }

  /*
  double esum = energy() + other.energy();

  double w1 = 0;
  double w2 = 0;

  //ignore negative energies
  if (energy() <= 0.)
    {
      if (other.energy()>0.)
        {
          //use the other hit + sum energy
          //setEnergy(other.energy());
          setEnergy(esum);
          setTime(other.time());
          setP(other.position());
        }
      else
        {
          //both are negative -> set both to 0
          setEnergy(0.);
          setTime(0.);
          setP(CLHEP::Hep3Vector(0,0,0));
          //both are negative -> set both to 0
        }
    }
  else if (other.energy() <0.)
    {
      //keep original, but with sum energy
      setEnergy(esum);
    }
  else if (esum > 0) {
    w1 =  energy()/esum;
    w2 =  other.energy()/esum;

    m_pos = w1*m_pos + w2*other.m_pos;
    setEnergy(esum);
    setTime(w1* time()+ w2 * other.time());
    //m_time = w1*m_time + w2*other.m_time; //average time??
    //what about m_ID...
  }
  else
    {
      std::cout <<"Wow, you're still here??"<<std::endl;
    }
  */
  return *this;
}
