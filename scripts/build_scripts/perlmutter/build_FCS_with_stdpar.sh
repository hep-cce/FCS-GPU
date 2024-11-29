#!/bin/bash

DPF_SCRATCH=/pscratch/sd/d/dingpf
ROOT_DIR=$DPF_SCRATCH/hep-mini-apps/root_install
SRC_DIR=$DPF_SCRATCH/hep-mini-apps/FCS-GPU_src

module load python
module load cmake/3.24.3
module load PrgEnv-nvhpc

source $ROOT_DIR/bin/thisroot.sh

WORK_DIR=$DPF_SCRATCH/hep-mini-apps-stdpar

BUILD_DIR=$WORK_DIR/FCS-GPU_build
INSTALL_DIR=$WORK_DIR/FCS-GPU_install

rm -rf $BUILD_DIR
mkdir -p $BUILD_DIR
mkdir -p $INSTALL_DIR
cd $BUILD_DIR

export CMAKE_PREFIX_PATH=/opt/nvidia/hpc_sdk/Linux_x86_64/23.9/cmake:$CMAKE_PREFIX_PATH
export LIBRARY_PATH=$LD_LIBRARY_PATH

cmake -DCMAKE_INSTALL_PREFIX=$INSTALL_DIR \
        -DENABLE_XROOTD=Off \
        -DCMAKE_CXX_STANDARD=17 \
        -DCMAKE_CXX_EXTENSIONS=Off \
        -DENABLE_GPU=on \
	-DUSE_STDPAR=ON \
	-DSTDPAR_TARGET=gpu \
        -DCMAKE_CUDA_ARCHITECTURES=80 \
        -DCMAKE_CXX_COMPILER=$SRC_DIR/scripts/nvc++_p \
        $SRC_DIR/FastCaloSimAnalyzer

make -j 16 install

