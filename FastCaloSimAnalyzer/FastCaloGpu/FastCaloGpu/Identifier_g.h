/*
  Copyright (C) 2002-2017 CERN for the benefit of the ATLAS collaboration
*/

#ifndef Identifier_GPU
#define Identifier_GPU

#ifndef IdentifierStandAlone
#define IdentifierStandAlone

#ifdef __HIPCC__
#define HIP_HOSTDEV __host__ __device__
#else
#define HIP_HOSTDEV
#endif

class Identifier
{
  public:
  typedef long long value_type;
  
  typedef enum bit_defs_enum
  {
    NBITS = sizeof(value_type) * 8, // bits per byte
    MAX_BIT = (static_cast<value_type>(1) << (NBITS - 1)),
    ALL_BITS = ~(static_cast<value_type>(0))
  } bit_defs;
  
  typedef enum max_value_type_enum {
    //max_value = 0xFFFFFFFFFFFFFFFFULL
    max_value = ~(static_cast<value_type>(0))
  } max_value_type;

HIP_HOSTDEV  Identifier():m_id(max_value) {};
HIP_HOSTDEV  Identifier(const Identifier& value):m_id(value.m_id) {};
HIP_HOSTDEV  Identifier(value_type value):m_id(value) {};
  
HIP_HOSTDEV  operator value_type() const { return m_id; }
  
HIP_HOSTDEV  Identifier& operator = (const Identifier& old) {m_id=old;return (*this);};
HIP_HOSTDEV  Identifier& operator = (value_type value) {m_id=value;return (*this);};
HIP_HOSTDEV  bool operator == (const Identifier& other) const {return (m_id == other.m_id);}
HIP_HOSTDEV  bool operator != (const Identifier& other) const {return (m_id != other.m_id);}
HIP_HOSTDEV  bool operator < (const Identifier& other) const {return (m_id < other.m_id);}
HIP_HOSTDEV  bool operator > (const Identifier& other) const {return (m_id > other.m_id);}
HIP_HOSTDEV  bool operator <= (const Identifier& other) const {return (m_id <= other.m_id);}
HIP_HOSTDEV  bool operator >= (const Identifier& other) const {return (m_id >= other.m_id);}
HIP_HOSTDEV  bool operator == (Identifier::value_type other) const {return (m_id == other);}
HIP_HOSTDEV  bool operator != (Identifier::value_type other) const {return (m_id != other);}
  
  protected:
  value_type m_id;
  
};
  
#endif
#endif
