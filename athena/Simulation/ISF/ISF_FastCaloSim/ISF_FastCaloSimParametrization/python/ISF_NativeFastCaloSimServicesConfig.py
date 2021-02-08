# Copyright (C) 2002-2017 CERN for the benefit of the ATLAS collaboration

"""
Tools configurations for ISF_NativeFastCaloSimServices
KG Tan, 04/12/2012
"""

from AthenaCommon.CfgGetter import getPrivateTool,getPrivateToolClone,getPublicTool,getPublicToolClone,\
        getService,getServiceClone,getAlgorithm,getAlgorithmClone

from AthenaCommon.Constants import *  # FATAL,ERROR etc.
from AthenaCommon.SystemOfUnits import *
from AthenaCommon.DetFlags import DetFlags

from ISF_Config.ISF_jobProperties import ISF_Flags # IMPORTANT: Flags must be set before tools are retrieved
from ISF_FastCaloSimParametrization.ISF_NativeFastCaloSimJobProperties import ISF_NativeFastCaloSimFlags

def getPunchThroughTool(name="ISF_PunchThroughTool", **kwargs):
    from G4AtlasApps.SimFlags import SimFlags,simFlags
    kwargs.setdefault("RandomNumberService"     , simFlags.RandomSvc()                               )
    kwargs.setdefault("RandomStreamName"        , ISF_FastCaloSimFlags.RandomStreamName()            )
    kwargs.setdefault("FilenameLookupTable"     , "CaloPunchThroughParametrisation.root"             )
    kwargs.setdefault("PunchThroughInitiators"  , [ 211 ]                                            )
    kwargs.setdefault("PunchThroughParticles"   , [    2212,     211,      22,      11,      13 ]    )
    kwargs.setdefault("DoAntiParticles"         , [   False,    True,   False,    True,    True ]    )
    kwargs.setdefault("CorrelatedParticle"      , [     211,    2212,      11,      22,       0 ]    )
    kwargs.setdefault("FullCorrelationEnergy"   , [ 100000., 100000., 100000., 100000.,      0. ]    )
    kwargs.setdefault("MinEnergy"               , [   938.3,   135.6,     50.,     50.,   105.7 ]    )
    kwargs.setdefault("MaxNumParticles"         , [      -1,      -1,      -1,      -1,      -1 ]    )
    kwargs.setdefault("EnergyFactor"            , [      1.,      1.,      1.,      1.,      1. ]    )
    kwargs.setdefault("BarcodeSvc"              , getService('ISF_LegacyBarcodeService')             )
    kwargs.setdefault("EnvelopeDefSvc"          , getService('AtlasGeometry_EnvelopeDefSvc')         )
    kwargs.setdefault("BeamPipeRadius"          , 500.						     )

    from ISF_PunchThroughTools.ISF_PunchThroughToolsConf import ISF__PunchThroughTool
    return ISF__PunchThroughTool(name, **kwargs )

def getEmptyCellBuilderTool(name="ISF_EmptyCellBuilderTool", **kwargs):
    from FastCaloSim.FastCaloSimConf import EmptyCellBuilderTool
    return EmptyCellBuilderTool(name, **kwargs )

def getFastHitConvertTool(name="ISF_FastHitConvertTool",**kwargs):
    from FastCaloSimHit.FastCaloSimHitConf import FastHitConvertTool 
    return FastHitConvertTool(name,**kwargs)

def getCaloNoiseTool(name="ISF_FCS_CaloNoiseTool", **kwargs):
    from CaloTools.CaloNoiseToolDefault import CaloNoiseToolDefault
    return CaloNoiseToolDefault(name, **kwargs )

def getAddNoiseCellBuilderTool(name="ISF_AddNoiseCellBuilderTool", **kwargs):
    kwargs.setdefault("CaloNoiseTool" , getPublicTool('ISF_FCS_CaloNoiseTool').getFullName())

    from FastCaloSim.FastCaloSimConf import AddNoiseCellBuilderTool
    return AddNoiseCellBuilderTool(name, **kwargs )

def getCaloCellContainerFinalizerTool(name="ISF_CaloCellContainerFinalizerTool", **kwargs):
    from CaloRec.CaloRecConf import CaloCellContainerFinalizerTool     
    return CaloCellContainerFinalizerTool(name, **kwargs )

#### NativeFastCaloSimSvc
def getNativeFastCaloSimSvc(name="ISF_NativeFastCaloSimSvc", **kwargs):
    from ISF_FastCaloSimParametrization.ISF_NativeFastCaloSimJobProperties import ISF_NativeFastCaloSimFlags
    kwargs.setdefault("BatchProcessMcTruth"              , False                                             )
    kwargs.setdefault("SimulateUndefinedBarcodeParticles", False                                             )
    kwargs.setdefault("Identifier"                       , 'NativeFastCaloSim'                                     )
    kwargs.setdefault("CaloCellsOutputName"              , ISF_NativeFastCaloSimFlags.CaloCellsName()              )
    kwargs.setdefault("PunchThroughTool"                 , getPublicTool('ISF_PunchThroughTool')             )
    kwargs.setdefault("DoPunchThroughSimulation"         , False                                             )
    kwargs.setdefault("ParticleBroker"                   , getService('ISF_ParticleBrokerSvc')               )
    kwargs.setdefault("CaloCellMakerTools_setup"         , [
                                                             getPublicTool('ISF_EmptyCellBuilderTool'),
                                                           ])
    kwargs.setdefault("CaloCellMakerTools_release"       , [
                                                             #getPublicTool('ISF_AddNoiseCellBuilderTool'),
                                                             getPublicTool('ISF_CaloCellContainerFinalizerTool'),
                                                             getPublicTool('ISF_FastHitConvertTool')
                                                           ])
    # let the ISF FCS flags know that FCS is being used
    ISF_NativeFastCaloSimFlags.NativeFastCaloSimIsActive.set_Value_and_Lock(True)

    # register the FastCaloSim random number streams
    from G4AtlasApps.SimFlags import SimFlags,simFlags
    simFlags.RandomSeedList.addSeed( ISF_NativeFastCaloSimFlags.RandomStreamName(), 98346412, 12461240 )

    from ISF_FastCaloSimParametrization.ISF_FastCaloSimParametrizationConf import ISF__NativeFastCaloSimSvc
    return ISF__NativeFastCaloSimSvc(name, **kwargs )

def getFastHitConvAlgFastCaloSimSvc(name="ISF_FastHitConvAlgFastCaloSimSvc",**kwargs):
    kwargs.setdefault("CaloCellMakerTools_release", [
                                                           #getPublicTool('ISF_AddNoiseCellBuilderTool'),
                                                            getPublicTool('ISF_CaloCellContainerFinalizerTool')
                                                    ] )
    # setup FastCaloSim hit converter and add it to the alg sequence:
    # -> creates HITS from reco cells
    from AthenaCommon.AlgSequence import AlgSequence
    topSequence=AlgSequence()
    topSequence+=getAlgorithm('ISF_FastHitConvAlg')
    return getNativeFastCaloSimSvc(name,**kwargs)

def getFastHitConvAlg(name="ISF_FastHitConvAlg", **kwargs):
    from ISF_FastCaloSimServices.ISF_FastCaloSimJobProperties import ISF_FastCaloSimFlags
    kwargs.setdefault("CaloCellsInputName"  , ISF_FastCaloSimFlags.CaloCellsName() )
    # TODO: do we need this?
    #from AthenaCommon.DetFlags import DetFlags
    #if DetFlags.pileup.LAr_on() or DetFlags.pileup.Tile_on():
    #  kwargs.setdefault("doPileup", True)
    #else:
    #  kwargs.setdefault("doPileup", False)
    from FastCaloSimHit.FastCaloSimHitConf import FastHitConv
    return FastHitConv(name, **kwargs )

