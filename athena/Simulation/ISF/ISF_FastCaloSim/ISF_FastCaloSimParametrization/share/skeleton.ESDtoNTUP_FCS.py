include("SimuJobTransforms/CommonSkeletonJobOptions.py")

# Get a handle to the ApplicationManager
from AthenaCommon.AppMgr import theApp
# Number of events to be processed (default is 10)
theApp.EvtMax = jobproperties.AthenaCommonFlags.EvtMax.get_Value()

# get the logger
from AthenaCommon.Logging import logging
fcsntuplog = logging.getLogger('FCS_Ntup_tf')
fcsntuplog.info( '****************** STARTING Ntuple Production *****************' )
fcsntuplog.info( str(runArgs) )

#==============================================================
# Job definition parameters:
#==============================================================
#already in CommonSkeletonJobOptions.py
#from AthenaCommon.AthenaCommonFlags import athenaCommonFlags
from AthenaCommon.AppMgr import ToolSvc
from AthenaCommon.AppMgr import ServiceMgr
import AthenaPoolCnvSvc.ReadAthenaPool

from PartPropSvc.PartPropSvcConf import PartPropSvc

import os
from glob import glob
if hasattr(runArgs,"inputESDFile"):
    globalflags.InputFormat.set_Value_and_Lock('pool')
    athenaCommonFlags.FilesInput = runArgs.inputESDFile
    ServiceMgr.EventSelector.InputCollections = athenaCommonFlags.FilesInput()
    pass


from GaudiSvc.GaudiSvcConf import THistSvc
ServiceMgr += THistSvc()
## Output NTUP_FCS File
if hasattr(runArgs,"outputNTUP_FCSFile"):
    print "Output is"
    print  runArgs.outputNTUP_FCSFile
    ServiceMgr.THistSvc.Output +=["ISF_HitAnalysis DATAFILE='"+runArgs.outputNTUP_FCSFile+"' OPT='RECREATE'"] # FIXME top level directory name
else:
    fcsntuplog.warning('No output file set')
    ServiceMgr.THistSvc.Output +=["ISF_HitAnalysis DATAFILE='output.NTUP_FCS.root' OPT='RECREATE'"] # FIXME top level directory name


## Optional output Geometry File
if hasattr(runArgs,"outputGeoFileName"):
    ServiceMgr.THistSvc.Output +=["ISF_Geometry DATAFILE='"+runArgs.outputGeoFileName+"' OPT='RECREATE'"] # FIXME top level directory name

## Flag for doG4Hits
if hasattr(runArgs,"doG4Hits"):
    doG4Hits = runArgs.doG4Hits
else:
    doG4Hits = False

## Flag for saveAllBranches
if hasattr(runArgs, "saveAllBranches"):
    saveAllBranches = runArgs.saveAllBranches
else:
    saveAllBranches = False

#==============================================================
# Job Configuration parameters:
#==============================================================
## Pre-exec
if hasattr(runArgs,"preExec"):
    fcsntuplog.info("transform pre-exec")
    for cmd in runArgs.preExec:
        fcsntuplog.info(cmd)
        exec(cmd)

## Pre-include
if hasattr(runArgs,"preInclude"):
    for fragment in runArgs.preInclude:
        include(fragment)


include("ISF_FastCaloSimParametrization/ISF_ntuple_core.py") # Main job options


## Post-include
if hasattr(runArgs,"postInclude"):
    for fragment in runArgs.postInclude:
        include(fragment)

## Post-exec
if hasattr(runArgs,"postExec"):
    fcsntuplog.info("transform post-exec")
    for cmd in runArgs.postExec:
        fcsntuplog.info(cmd)
        exec(cmd)
