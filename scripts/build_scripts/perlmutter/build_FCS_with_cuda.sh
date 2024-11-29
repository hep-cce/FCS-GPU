#!/bin/bash

# load the default python module, ROOT was built with it.

module load python

WORK_DIR=$SCRATCH/hep-mini-apps

SRC_DIR=$WORK_DIR/FCS-GPU_src
BUILD_DIR=$WORK_DIR/FCS-GPU_gpu_build
INSTALL_DIR=$WORK_DIR/FCS-GPU_gpu_install

ROOT_DIR=$WORK_DIR/root_install
source $ROOT_DIR/bin/thisroot.sh

cd $WORK_DIR
rm -rf $BUILD_DIR
mkdir -p $BUILD_DIR
mkdir -p $INSTALL_DIR
git clone https://github.com/cgleggett/FCS-GPU.git -b dingpf/packaging $SRC_DIR

cd $BUILD_DIR
cmake -DCMAKE_INSTALL_PREFIX=$INSTALL_DIR \
	-DENABLE_XROOTD=Off \
	-DCMAKE_CXX_STANDARD=17 \
	-DCMAKE_CXX_EXTENSIONS=Off \
       	-DENABLE_GPU=on \
	-DCMAKE_CUDA_ARCHITECTURES=80 \
	$SRC_DIR/FastCaloSimAnalyzer  

make -j 128 install
