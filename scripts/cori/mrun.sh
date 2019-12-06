#!/bin/bash

# Execute a number of runs of FastCaloSim
# should be called with "srun -n1 -N1 ..."

# enable nvidia-cuda-mps-control
USE_MPS_SERVER=1
# log output of nvidia-smi every N seconds if N>0
SMI_INTERVAL=0

if [[ "$#" -eq 0 ]]; then
    N1=1
    N2=$( lscpu | grep '^CPU(s):' | awk '{print $NF}' )
elif [[ "$#" -eq 1 ]]; then
    N1=$1
    N2=$1
elif [[ "$#" -eq 2 ]]; then
    N1=$1
    N2=$2
else
    echo "ERROR: bad arguments"
    exit 1
fi

# echo "N1: $N1  N2: $N2"

SMI_PID=0

export CUDA_MPS_LOG_DIRECTORY="/tmp/${USER}/cuda-mps/logs"
mkdir -p $CUDA_MPS_LOG_DIRECTORY
export CUDA_VISIBLE_DEVICES="0"
if [[ $USE_MPS_SERVER -eq 1 ]]; then
    echo "starting mps server"
    nvidia-cuda-mps-control -d
fi

SHM_DATA_DIR="/dev/shm/${USER}/FastCaloSimInputs"
if [[ ! -d $SHM_DATA_DIR ]]; then
    echo "creating shm dir and copying input files"
    mkdir -p $SHM_DATA_DIR
    cp -rp /global/project/projectdirs/atlas/leggett/data/FastCaloSimInputs/* $SHM_DATA_DIR
    echo "done"
else
    echo "input files already in $SHM_DATA_DIR"
fi

if [[ $SMI_INTERVAL -ne 0 ]]; then
    echo "saving nvidia-smi"
    nvidia-smi -f smi.log --id=$CUDA_VISIBLE_DEVICES -l $SMI_INTERVAL -q --display=MEMORY,PIDS &
    SMI_PID=$!
fi

for n in `seq $N1 $N2`; do
    ./run.sh $n
    if [[ $? -ne 0 ]]; then
        exit 1
    fi
    sleep 1
done

if [[ $USE_MPS_SERVER -eq 1 ]]; then
    echo quit | nvidia-cuda-mps-control
fi

if [[ $SMI_PID -ne 0 ]]; then
    sleep 15
    kill -9 $SMI_PID
fi
