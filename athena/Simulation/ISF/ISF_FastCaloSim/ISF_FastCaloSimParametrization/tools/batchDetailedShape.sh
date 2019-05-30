#!/bin/bash
#BSUB -q 8nh
#BSUB -J AF2[1-1]
#BSUB -o LOGS/out
#BSUB -e LOGS/err
#BSUB -g /AF2
#BSUB ulimit -c 0
  
cd /afs/cern.ch/work/c/conti/private/G4FASTCALO/ISF_FastCaloSimParametrization/tools/

RELEASE=20.1.2
source /afs/cern.ch/atlas/software/dist/AtlasSetup/scripts/asetup.sh $RELEASE

let "s=($LSB_JOBINDEX)"
echo $s

if [ $s -eq 1 ]; then 
    root -l -b runDetailedShape.C
fi

