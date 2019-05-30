#!/bin/bash

dsid=$1
dsid_zv0=$2
do2Dparameterization=$3
isPhisymmetry=$4
doMeanRz=$5
useMeanRz=$6
doZVertexStudies=$7

cd macro
root -l -x -q -b "runZVertexStudies.C($dsid,$dsid_zv0,$do2Dparameterization,$isPhisymmetry,$doMeanRz,$useMeanRz,$doZVertexStudies)"
rc=$?
exit $rc
