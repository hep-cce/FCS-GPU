# Copyright (C) 2002-2017 CERN for the benefit of the ATLAS collaboration


"""
Tools configurations for ISF_FastCaloSimParametrization
"""
from AthenaCommon import CfgMgr
from AthenaCommon.Constants import *  # FATAL,ERROR etc.
from AthenaCommon.SystemOfUnits import *
from AthenaCommon.DetFlags import DetFlags

def getFastCaloSimCaloExtrapolation(name="FastCaloSimCaloExtrapolation", **kwargs):
    from ISF_FastCaloSimParametrization.ISF_FastCaloSimParametrizationConf import FastCaloSimCaloExtrapolation

    kwargs.setdefault("CaloBoundaryR"             , [1148.0, 120.0, 41.0] )
    kwargs.setdefault("CaloBoundaryZ"             , [3550.0, 4587.0, 4587.0] )
    kwargs.setdefault("CaloMargin"                , 100    )
    kwargs.setdefault("Extrapolator"              , "TimedExtrapolator" )
    kwargs.setdefault("CaloSurfaceHelper"         , "CaloSurfaceHelper" )
    kwargs.setdefault("CaloGeometryHelper"        , "FastCaloSimGeometryHelper" )
    kwargs.setdefault("CaloEntrance"              , "InDet::Containers::InnerDetector"     )
    
    return CfgMgr.FastCaloSimCaloExtrapolation(name, **kwargs)

def getFastCaloSimGeometryHelper(name="FastCaloSimGeometryHelper", **kwargs):
    from ISF_FastCaloSimParametrization.ISF_FastCaloSimParametrizationConf import FastCaloSimGeometryHelper
    return CfgMgr.FastCaloSimGeometryHelper(name, **kwargs)
