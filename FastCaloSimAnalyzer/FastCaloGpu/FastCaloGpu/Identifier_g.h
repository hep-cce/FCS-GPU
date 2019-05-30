/*
  Copyright (C) 2002-2017 CERN for the benefit of the ATLAS collaboration
*/

#ifndef Identifier_GPU
#define Identifier_GPU

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

__host__ __device__  Identifier():m_id(max_value) {};
__host__ __device__  Identifier(const Identifier& value):m_id(value.m_id) {};
__host__ __device__  Identifier(value_type value):m_id(value) {};
  
__host__ __device__  operator value_type() const { return m_id; }
  
__host__ __device__  Identifier& operator = (const Identifier& old) {m_id=old;return (*this);};
__host__ __device__  Identifier& operator = (value_type value) {m_id=value;return (*this);};
__host__ __device__  bool operator == (const Identifier& other) const {return (m_id == other.m_id);}
__host__ __device__  bool operator != (const Identifier& other) const {return (m_id != other.m_id);}
__host__ __device__  bool operator < (const Identifier& other) const {return (m_id < other.m_id);}
__host__ __device__  bool operator > (const Identifier& other) const {return (m_id > other.m_id);}
__host__ __device__  bool operator <= (const Identifier& other) const {return (m_id <= other.m_id);}
__host__ __device__  bool operator >= (const Identifier& other) const {return (m_id >= other.m_id);}
__host__ __device__  bool operator == (Identifier::value_type other) const {return (m_id == other);}
__host__ __device__  bool operator != (Identifier::value_type other) const {return (m_id != other);}
  
  protected:
  value_type m_id;
  
};
  
#endif

