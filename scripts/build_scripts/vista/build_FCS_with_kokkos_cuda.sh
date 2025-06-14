#!/bin/bash

ROOT_DIR=$WORK/hep-mini-apps/root_install
SRC_DIR=$WORK/hep-mini-apps/FCS-GPU_src
KOKKOS_DIR=$WORK/hep-mini-apps-kokkos/kokkos_install

#module load python
#module load cmake/3.24.3


ml cuda/12.5
ml gcc/13.2.0

WORK_DIR=$WORK/hep-mini-apps-kokkos

BUILD_DIR=$WORK_DIR/FCS-GPU_build
INSTALL_DIR=$WORK_DIR/FCS-GPU_install

source $ROOT_DIR/bin/thisroot.sh

rm -rf $BUILD_DIR
mkdir -p $BUILD_DIR
mkdir -p $INSTALL_DIR
cd $BUILD_DIR

export CMAKE_PREFIX_PATH=$KOKKOS_DIR:$CMAKE_PREFIX_PATH

export CC=/opt/apps/gcc/13.2.0/bin/gcc
export CXX=/opt/apps/gcc/13.2.0/bin/g++

cmake -DCMAKE_INSTALL_PREFIX=$INSTALL_DIR \
        -DENABLE_XROOTD=Off \
        -DCMAKE_CXX_EXTENSIONS=Off \
        -DENABLE_GPU=on \
        -DCMAKE_CUDA_ARCHITECTURES=90 \
        -DUSE_KOKKOS=ON \
        -DCMAKE_CXX_COMPILER=$KOKKOS_DIR/bin/nvcc_wrapper \
        -DCMAKE_CXX_STANDARD=17 \
        $SRC_DIR/FastCaloSimAnalyzer

make -j 16 install

echo "Run the following before calling the application"
echo "module load python"
echo "source $ROOT_DIR/bin/thisroot.sh"
echo "export FCS_DATAPATH=/global/cfs/cdirs/atlas/leggett/data/FastCaloSimInputs"
echo "export LD_LIBRARY_PATH=$KOKKOS_DIR/lib64:$LD_LIBRARY_PATH"
echo "source $INSTALL_DIR/setup.sh"
