# Copyright (C) 2002-2019 CERN for the benefit of the ATLAS collaboration

# Enable XRootD support
set(ENABLE_XROOTD OFF CACHE BOOL "Enable XRootD support")


if (ENABLE_XROOTD)
  find_package(XRootD REQUIRED)

  list(APPEND FCS_CommonDefinitions -DENABLE_XROOTD_SUPPORT)
endif()
