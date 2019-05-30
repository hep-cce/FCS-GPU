# Copyright (C) 2002-2017 CERN for the benefit of the ATLAS collaboration

## @file ISF_NativeFastCaloSimJobProperties.py
## @purpose Python module to hold common flags to configure JobOptions
##

""" ISF_NativeFastCaloSimJobProperties
    Python module to hold storegate keys of InDet objects.

"""

__author__ = "KG Tan"
__version__= "$Revision: 779694 $"
__doc__    = "ISF_NativeFastCaloSimJobProperties"

__all__    = [ "ISF_NativeFastCaloSimJobProperties" ]

# kindly stolen from AthenaCommonFlags from S. Binet and M. Gallas

##-----------------------------------------------------------------------------
## Import

from AthenaCommon.JobProperties import JobProperty, JobPropertyContainer
from AthenaCommon.JobProperties import jobproperties

##-----------------------------------------------------------------------------
## Define the flag

class NativeFastCaloSimIsActive(JobProperty):
    """Defines whether or not NativeFastCaloSim is being run in the current athena setup"""
    statusOn     = True
    allowedTypes = ['bool']
    StoredValue  = False

class RandomStreamName(JobProperty):
    """The random number stream used by FastCaloSim"""
    statusOn     = True
    allowedTypes = ['str']
    StoredValue  = 'FastCaloSimRnd'

class CaloCellsName(JobProperty):
    """StoreGate collection name for FastCaloSim hits"""
    statusOn     = True
    allowedTypes = ['str']
    StoredValue  = 'AllCalo'

class outputFile(JobProperty):
    statusOn     = False
    allowedTypes = ['str']
    StoredValue  = 'ESD_output_test.root'

##-----------------------------------------------------------------------------
## 2nd step
## Definition of the InDet flag container
class ISF_NativeFastCaloSimJobProperties(JobPropertyContainer):
    """Container for the ISF_FastCaloSim key flags
    """
    pass

##-----------------------------------------------------------------------------
## 3rd step
## adding the container to the general top-level container
jobproperties.add_Container(ISF_NativeFastCaloSimJobProperties)


##-----------------------------------------------------------------------------
## 4th step
## adding the flags to the  container
jobproperties.ISF_NativeFastCaloSimJobProperties.add_JobProperty( NativeFastCaloSimIsActive        )
jobproperties.ISF_NativeFastCaloSimJobProperties.add_JobProperty( RandomStreamName           )
jobproperties.ISF_NativeFastCaloSimJobProperties.add_JobProperty( CaloCellsName              )
jobproperties.ISF_NativeFastCaloSimJobProperties.add_JobProperty( outputFile              )
##-----------------------------------------------------------------------------
## 5th step
## short-cut for lazy people
## carefull: do not select ISF_FastCaloSimJobProperties as a short name as well. 
## otherwise problems with pickle
## Note: you still have to import it:
ISF_NativeFastCaloSimFlags = jobproperties.ISF_NativeFastCaloSimJobProperties
