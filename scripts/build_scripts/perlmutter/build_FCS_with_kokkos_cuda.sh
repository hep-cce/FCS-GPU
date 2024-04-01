#!/bin/bash

ROOT_DIR=$SCRATCH/hep-mini-apps/root_install
SRC_DIR=$SCRATCH/hep-mini-apps/FCS-GPU_src
KOKKOS_DIR=$SCRATCH/hep-mini-apps-kokkos/kokkos_install

module load python
module load cmake/3.24.3

source $ROOT_DIR/bin/thisroot.sh

WORK_DIR=$SCRATCH/hep-mini-apps-kokkos

BUILD_DIR=$WORK_DIR/FCS-GPU_build
INSTALL_DIR=$WORK_DIR/FCS-GPU_install

rm -rf $BUILD_DIR
mkdir -p $BUILD_DIR
mkdir -p $INSTALL_DIR
cd $BUILD_DIR

export CMAKE_PREFIX_PATH=$KOKKOS_DIR:$CMAKE_PREFIX_PATH

cmake -DCMAKE_INSTALL_PREFIX=$INSTALL_DIR \
        -DENABLE_XROOTD=Off \
        -DCMAKE_CXX_EXTENSIONS=Off \
        -DENABLE_GPU=on \
        -DCMAKE_CUDA_ARCHITECTURES=80 \
        -DUSE_KOKKOS=ON \
        -DCMAKE_CXX_COMPILER=$KOKKOS_DIR/bin/nvcc_wrapper \
        -DCMAKE_CXX_STANDARD=17 \
        $SRC_DIR/FastCaloSimAnalyzer

make -j 16 install

echo "Run the following before calling the application"
echo "module load python"
echo "source $ROOT_DIR/bin/thisroot.sh"
echo "export FCS_DATAPATH=/global/cfs/cdirs/atlas/leggett/data/FastCaloSimInputs"
echo "export LD_LIBRARY_PATH=$KOKKOS_DIR/lib:$LD_LIBRARY_PATH"
echo "source $INSTALL_DIR/setup.sh"
