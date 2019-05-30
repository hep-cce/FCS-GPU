# production options for running ISF ntuples
from AthenaCommon.AppMgr import ServiceMgr
#import AthenaPoolCnvSvc.ReadAthenaPool
#from PartPropSvc.PartPropSvcConf import PartPropSvc
#include( "ParticleBuilderOptions/McAOD_PoolCnv_jobOptions.py")
#include( "EventAthenaPool/EventAthenaPool_joboptions.py" )

from AthenaCommon.AlgSequence import AlgSequence
topSequence = AlgSequence()

from ISF_FastCaloSimParametrization.ISF_NativeFastCaloSimJobProperties import jobproperties
from OutputStreamAthenaPool.MultipleStreamManager import MSMgr
alg = MSMgr.NewRootStream( "StreamNTUP_FastCaloSim", jobproperties.ISF_NativeFastCaloSimJobProperties.outputFile(), "FastCaloSim" )

from ISF_FastCaloSimParametrization.ISF_FastCaloSimParametrizationConf import ISF_HitAnalysis
alg += ISF_HitAnalysis() 

ISF_HitAnalysis = ISF_HitAnalysis()
ISF_HitAnalysis.NtupleFileName = 'ISF_HitAnalysis'

##############################
#ISF_HitAnalysis.CaloBoundaryR = [ 0., 1148., 1148., 0. ]
#ISF_HitAnalysis.CaloBoundaryZ = [ -3475., -3475., 3475., 3475. ]
ISF_HitAnalysis.CaloBoundaryR = 1148.0
ISF_HitAnalysis.CaloBoundaryZ = 3549.5 #before: 3475.0
ISF_HitAnalysis.CaloMargin=100 #=10cm
ISF_HitAnalysis.NTruthParticles = 1 # Copy only one truth particle to the ntuples for now
#ISF_HitAnalysis.OutputLevel = DEBUG

#############################
##### NEW TRACKING SETUP ####
#############################
from TrkExSTEP_Propagator.TrkExSTEP_PropagatorConf import Trk__STEP_Propagator
niPropagator = Trk__STEP_Propagator()
niPropagator.MaterialEffects = False 
ToolSvc+=niPropagator    

from TrkExTools.TimedExtrapolator import TimedExtrapolator
timedExtrapolator=TimedExtrapolator()
timedExtrapolator.STEP_Propagator = niPropagator
timedExtrapolator.ApplyMaterialEffects = False
ToolSvc+=timedExtrapolator

from CaloTrackingGeometry.CaloTrackingGeometryConf import CaloSurfaceHelper
caloSurfaceHelper = CaloSurfaceHelper()
ToolSvc+=caloSurfaceHelper

from TrkDetDescrSvc.TrkDetDescrJobProperties import TrkDetFlags 

ISF_HitAnalysis.CaloEntrance=TrkDetFlags.InDetContainerName()
ISF_HitAnalysis.CaloSurfaceHelper=caloSurfaceHelper
ISF_HitAnalysis.Extrapolator=timedExtrapolator

#############################

from ISF_FastCaloSimParametrization.ISF_FastCaloSimParametrizationConf import FastCaloSimGeometryHelper
FCSgeoHelper=FastCaloSimGeometryHelper()
ToolSvc+=FCSgeoHelper
ISF_HitAnalysis.CaloGeometryHelper=FCSgeoHelper

# Note: don't think I actually need ISF_Geometry as a stream
# Note: Need to turn ISF_HitAnalysis into a D3PD friendly object

