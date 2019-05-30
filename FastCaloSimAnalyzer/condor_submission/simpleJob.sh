#!/bin/bash

echo \"-----------------------\"
echo -n \"The job started at : \" ; date
echo \"-----------------------\"
echo

echo "parameters of the job are: $*"
  FastCaloSimAnalyzerPath=$1
  command=$2
  parameters=(${*:3})
      
echo "Running on the host: `hostname`"

echo "base dir is: " $FastCaloSimAnalyzerPath
echo "Command is: " $command
echo "Parameters: " ${parameters[@]}


echo "Current directory: `pwd`"

cd $FastCaloSimAnalyzerPath

echo "Current directory: `pwd`"

echo "Setting up the environment"

export ATLAS_LOCAL_ROOT_BASE=/cvmfs/atlas.cern.ch/repo/ATLASLocalRootBase
source $ATLAS_LOCAL_ROOT_BASE/user/atlasLocalSetup.sh
source $ATLAS_LOCAL_ROOT_BASE/packageSetups/localSetup.sh ROOT

cmd="$command ${parameters[@]}"
echo "Running $cmd"
$cmd 


rc=$?

echo
echo "-----------------------"
echo -n "The job ended with rc= $rc at : " ; date
echo "-----------------------"
echo 

exit $rc
