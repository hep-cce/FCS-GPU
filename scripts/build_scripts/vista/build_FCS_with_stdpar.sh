#!/bin/bash

ROOT_DIR=$WORK/hep-mini-apps/root_install
SRC_DIR=$WORK/hep-mini-apps/FCS-GPU_src

ml nvidia/24.7

source $ROOT_DIR/bin/thisroot.sh

WORK_DIR=$WORK/hep-mini-apps-stdpar

BUILD_DIR=$WORK_DIR/FCS-GPU_build
INSTALL_DIR=$WORK_DIR/FCS-GPU_install

rm -rf $BUILD_DIR
mkdir -p $BUILD_DIR
mkdir -p $INSTALL_DIR
cd $BUILD_DIR

export CMAKE_PREFIX_PATH=/home1/apps/nvidia/Linux_aarch64/24.7/cmake:$CMAKE_PREFIX_PATH
export LIBRARY_PATH=$LD_LIBRARY_PATH

cmake -DCMAKE_INSTALL_PREFIX=$INSTALL_DIR \
        -DENABLE_XROOTD=Off \
        -DCMAKE_CXX_STANDARD=17 \
        -DCMAKE_CXX_EXTENSIONS=Off \
        -DENABLE_GPU=on \
	-DUSE_STDPAR=ON \
	-DSTDPAR_TARGET=gpu \
        -DCMAKE_CUDA_ARCHITECTURES=90 \
        -DCMAKE_CXX_COMPILER=$SRC_DIR/scripts/nvc++_p \
        $SRC_DIR/FastCaloSimAnalyzer

make -j 16 install

