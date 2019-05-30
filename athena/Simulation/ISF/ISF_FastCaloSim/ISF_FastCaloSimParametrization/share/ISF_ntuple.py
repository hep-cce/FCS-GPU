from AthenaCommon.AppMgr import ServiceMgr
import AthenaPoolCnvSvc.ReadAthenaPool

from PartPropSvc.PartPropSvcConf import PartPropSvc

include( "ParticleBuilderOptions/McAOD_PoolCnv_jobOptions.py")
include( "EventAthenaPool/EventAthenaPool_joboptions.py" )

import os
import sys
from glob import glob
from AthenaCommon.AthenaCommonFlags  import athenaCommonFlags
#specify input file here
athenaCommonFlags.FilesInput = ["/afs/cern.ch/work/a/ahasib/public/photon.50GeV.ESD.pool.root"]

doG4Hits = False
saveAllBranches = False

include("ISF_FastCaloSimParametrization/ISF_ntuple_core.py")

theApp.EvtMax = 100 # Set to -1 for all events

from GaudiSvc.GaudiSvcConf import THistSvc
ServiceMgr += THistSvc()
#name the output file here
OutputName="ESD_output_test.root"
OutputName=OutputName.replace("ESD","ISF_HitAnalysis")
print OutputName
#Use this to automatically name the output file (rename ESD->ISF_HitAnalysis)
ServiceMgr.THistSvc.Output += [ "ISF_HitAnalysis DATAFILE='"+OutputName+"' OPT='RECREATE'" ]
from AthenaCommon.GlobalFlags import jobproperties
# ServiceMgr.THistSvc.Output += [ "ISF_Geometry DATAFILE='output_geo.root' OPT='RECREATE'" ]

