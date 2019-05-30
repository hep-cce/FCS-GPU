#! /usr/bin/env python

# Copyright (C) 2002-2017 CERN for the benefit of the ATLAS collaboration

"""
Run HITS file and produce histograms.
"""

import os.path
import sys
import time
import logging

# Setup core logging here
from PyJobTransforms.trfLogger import msg
msg.info('logging set in %s' % sys.argv[0])
from PyJobTransforms.transform import transform
from PyJobTransforms.trfExe import athenaExecutor
from PyJobTransforms.trfArgs import addAthenaArguments, addDetectorArguments, addTriggerArguments
from PyJobTransforms.trfDecorators import stdTrfExceptionHandler, sigUsrStackTrace
import PyJobTransforms.trfArgClasses as trfArgClasses
from ISF_FastCaloSimParametrization.fcsTrfArgs import addFCS_NtupArgs

@stdTrfExceptionHandler
@sigUsrStackTrace
def main():

    msg.info('This is %s' % sys.argv[0])

    trf = getTransform()
    trf.parseCmdLineArgs(sys.argv[1:])
    trf.execute()
    trf.generateReport()

    msg.info("%s stopped at %s, trf exit code %d" % (sys.argv[0], time.asctime(), trf.exitCode))
    sys.exit(trf.exitCode)

def getTransform():
    executorSet = set()
    executorSet.add(athenaExecutor(name = 'FCS_Ntup',
                                   skeletonFile = 'ISF_FastCaloSimParametrization/skeleton.ESDtoNTUP_FCS.py',
                                   inData = ['ESD'], outData = ['NTUP_FCS'],))
    trf = transform(executor = executorSet, description = 'FastCaloSim V2 Parametrization ntuple transform. Inputs must be ESD. Outputs must bentuple files.')
    addAthenaArguments(trf.parser)
    addFCS_NtupArgs(trf.parser)
    return trf


if __name__ == '__main__':
    main()
