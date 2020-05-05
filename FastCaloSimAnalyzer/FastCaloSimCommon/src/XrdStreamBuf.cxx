/*
  Copyright (C) 2002-2019 CERN for the benefit of the ATLAS collaboration
*/

#include "XrdStreamBuf.h"

XrdStreamBuf::XrdStreamBuf( const std::string& fileUrl, uint32_t bufferSize )
    : _buffer( new char[bufferSize] ), _bufferSize( bufferSize ), _file( std::make_unique<XrdCl::File>() ) {
  auto status = _file->Open( fileUrl, XrdCl::OpenFlags::Read );
  if ( !status.IsOK() ) { throw std::runtime_error( status.ToString() ); }
}

XrdStreamBuf::~XrdStreamBuf() {
  delete[] _buffer;
  auto status = _file->Close();
  if ( !status.IsOK() ) { std::cout << status.ToString() << std::endl; }
}

int XrdStreamBuf::underflow() {
  if ( gptr() == egptr() ) {
    uint32_t bytesRead;
    auto     status = _file->Read( _totalRead, _bufferSize, _buffer, bytesRead );
    if ( !status.IsOK() ) { throw std::runtime_error( status.ToString() ); }
    setg( _buffer, _buffer, _buffer + bytesRead );
    _totalRead += bytesRead;
  }

  return gptr() == egptr() ? std::char_traits<char>::eof() : std::char_traits<char>::to_int_type( *gptr() );
}
