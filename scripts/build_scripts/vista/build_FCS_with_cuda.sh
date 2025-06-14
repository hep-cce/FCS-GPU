#!/bin/bash

# load the default python module, ROOT was built with it.

#module load python
ml cuda/12.5
ml gcc/13.2.0

WORK_DIR=$WORK/hep-mini-apps

SRC_DIR=$WORK_DIR/FCS-GPU_src
BUILD_DIR=$WORK_DIR/FCS-GPU_gpu_build
INSTALL_DIR=$WORK_DIR/FCS-GPU_gpu_install

ROOT_DIR=$WORK_DIR/root_install
source $ROOT_DIR/bin/thisroot.sh

cd $WORK_DIR
rm -rf $BUILD_DIR
mkdir -p $BUILD_DIR
mkdir -p $INSTALL_DIR
#git clone https://github.com/hep-cce/FCS-GPU.git -b dingpf/packaging $SRC_DIR

cd $BUILD_DIR
export CC=/opt/apps/gcc/13.2.0/bin/gcc
export CXX=/opt/apps/gcc/13.2.0/bin/g++
#export CXXFLAGS="-z common-page-size=0x10000 -z max-page-size=0x10000 "

cmake -DCMAKE_INSTALL_PREFIX=$INSTALL_DIR \
	-DENABLE_XROOTD=Off \
	-DCMAKE_CXX_STANDARD=17 \
	-DCMAKE_CXX_EXTENSIONS=Off \
       	-DENABLE_GPU=on \
	-DCMAKE_CUDA_ARCHITECTURES=90 \
	$SRC_DIR/FastCaloSimAnalyzer  
#export NVCC_PREPEND_FLAGS='-ccbin /opt/apps/gcc/13.2.0/bin/g++'

make -j 32 install
