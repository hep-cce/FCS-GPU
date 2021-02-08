# Copyright (C) 2002-2017 CERN for the benefit of the ATLAS collaboration

"""
Configuration database for ISF_FastCaloSimParametrization
"""

from AthenaCommon.CfgGetter import addTool

from AthenaCommon.Constants import *  # FATAL,ERROR etc.
import AthenaCommon.SystemOfUnits as Units

# Common tools, services and algorithms used by jobs
addTool("ISF_FastCaloSimParametrization.ISF_FastCaloSimParametrizationConfig.getFastCaloSimCaloExtrapolation" , "FastCaloSimCaloExtrapolation"   )
addTool("ISF_FastCaloSimParametrization.ISF_FastCaloSimParametrizationConfig.getFastCaloSimGeometryHelper" , "FastCaloSimGeometryHelper"   )
