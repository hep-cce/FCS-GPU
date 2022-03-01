/*
  Copyright (C) 2002-2017 CERN for the benefit of the ATLAS collaboration
*/

#ifndef Identifier_GPU
#define Identifier_GPU

#ifndef IdentifierStandAlone
#  define IdentifierStandAlone

#include "HostDevDef.h"

class Identifier {
public:
  typedef long long value_type;

  typedef enum bit_defs_enum {
    NBITS    = sizeof( value_type ) * 8, // bits per byte
    MAX_BIT  = ( static_cast<value_type>( 1 ) << ( NBITS - 1 ) ),
    ALL_BITS = ~( static_cast<value_type>( 0 ) )
  } bit_defs;

  typedef enum max_value_type_enum {
    // max_value = 0xFFFFFFFFFFFFFFFFULL
    max_value = ~( static_cast<value_type>( 0 ) )
  } max_value_type;

  __HOSTDEV__ Identifier() : m_id( max_value ){};
  __HOSTDEV__ Identifier( const Identifier& value ) : m_id( value.m_id ){};
  __HOSTDEV__ Identifier( value_type value ) : m_id( value ){};

  __HOSTDEV__ operator value_type() const { return m_id; }

  __HOSTDEV__ Identifier& operator=( const Identifier& old ) {
    m_id = old;
    return ( *this );
  };
  __HOSTDEV__ Identifier& operator=( value_type value ) {
    m_id = value;
    return ( *this );
  };
  __HOSTDEV__ bool operator==( const Identifier& other ) const { return ( m_id == other.m_id ); }
  __HOSTDEV__ bool operator!=( const Identifier& other ) const { return ( m_id != other.m_id ); }
  __HOSTDEV__ bool operator<( const Identifier& other ) const { return ( m_id < other.m_id ); }
  __HOSTDEV__ bool operator>( const Identifier& other ) const { return ( m_id > other.m_id ); }
  __HOSTDEV__ bool operator<=( const Identifier& other ) const { return ( m_id <= other.m_id ); }
  __HOSTDEV__ bool operator>=( const Identifier& other ) const { return ( m_id >= other.m_id ); }
  __HOSTDEV__ bool operator==( Identifier::value_type other ) const { return ( m_id == other ); }
  __HOSTDEV__ bool operator!=( Identifier::value_type other ) const { return ( m_id != other ); }

protected:
  value_type m_id;
};

#endif
#endif
