/*
  Copyright (C) 2002-2019 CERN for the benefit of the ATLAS collaboration
*/

#ifndef XRDSTREAMBUF_H
#define XRDSTREAMBUF_H

#include <iostream>
#include <memory>

#include <XrdCl/XrdClFile.hh>

class XrdStreamBuf : public std::streambuf {
public:
  XrdStreamBuf( const std::string& fileUrl, uint32_t bufferSize = 4 * 1024 * 1024 );
  ~XrdStreamBuf();

  virtual int underflow() final;

private:
  uint64_t _totalRead{};
  char*    _buffer;
  uint32_t _bufferSize;

  std::unique_ptr<XrdCl::File> _file{};
};

#endif // XRDSTREAMBUF_H
