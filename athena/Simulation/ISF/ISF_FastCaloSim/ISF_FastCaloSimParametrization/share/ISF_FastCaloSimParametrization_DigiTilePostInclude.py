from AthenaCommon.AppMgr import ServiceMgr

ToolSvc.TileHitVecToCntTool.HitTimeFlag=1

ServiceMgr.TileInfoLoader.NPhElec=0
ServiceMgr.TileInfoLoader.NPhElecA=0
ServiceMgr.TileInfoLoader.NPhElecC=0
ServiceMgr.TileInfoLoader.NPhElecD=0
ServiceMgr.TileInfoLoader.NPhElecE=0

topSequence.TileDigitsMaker.IntegerDigits=True

###########
print ServiceMgr.TileInfoLoader
print topSequence.TileDigitsMaker
print ToolSvc.TileHitVecToCntTool
#topSequence.TileDigitsMaker.OutputLevel=1
#TileHitVecToCntTool.OutputLevel=1
#ToolSvc.TileRawChannelBuilderOptATLAS.OutputLevel=1
#ServiceMgr.MessageSvc.OutputLevel= VERBOSE
###########
