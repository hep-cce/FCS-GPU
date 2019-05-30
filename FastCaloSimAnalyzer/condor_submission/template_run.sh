#!/bin/bash
echo " Running in: $PWD  @ $HOSTNAME"
echo "Time : " $(date -u)

echo "Setup ROOT ..."
export ATLAS_LOCAL_ROOT_BASE=/cvmfs/atlas.cern.ch/repo/ATLASLocalRootBase
source $ATLAS_LOCAL_ROOT_BASE/user/atlasLocalSetup.sh
lsetup "root 6.14.04-x86_64-slc6-gcc62-opt"

echo "cd to the correct directory"
cd @SUBMIT_DIR@
echo "current directory: " $PWD

echo "Running energy parametrization: " $(date -u)
time root -b -q runEpara.C
echo "Running MeanRZ : " $(date -u)
time root -b -q runMeanRZ.C
echo "Running shape parametrization:" $(date -u)
time root -b -q runShape.C
echo "End of run:" $(date -u)

