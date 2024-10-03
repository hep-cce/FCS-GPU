#!/bin/bash

module load python
module load cmake/3.24.3

INSTALL_DIR=$SCRATCH/hep-mini-apps-kokkos/kokkos_install
SRC_DIR=$SCRATCH/hep-mini-apps-kokkos/kokkos
BUILD_DIR=$SCRATCH/hep-mini-apps-kokkos/kokkos_build

mkdir -p $INSTALL_DIR
mkdir -p $BUILD_DIR

pushd $BUILD_DIR

cmake -DCMAKE_INSTALL_PREFIX=${INSTALL_DIR} \
      -DCMAKE_CXX_COMPILER=${SRC_DIR}/bin/nvcc_wrapper \
      -DCMAKE_CXX_STANDARD=17 \
      -DCMAKE_CXX_EXTENSIONS=Off \
      -DKokkos_ENABLE_COMPILER_WARNINGS=ON \
      -DKokkos_ENABLE_CUDA=ON \
      -DKokkos_ENABLE_CUDA_LAMBDA=ON \
      -DKokkos_ENABLE_CUDA_RELOCATABLE_DEVICE_CODE=Off \
      -DKokkos_ENABLE_OPENMP=On \
      -DKokkos_ENABLE_SERIAL=On \
      -DKokkos_ENABLE_TESTS=Off \
      -DKokkos_ARCH_AMPERE80=ON \
      -DBUILD_SHARED_LIBS=ON \
      ${SRC_DIR}

make -j 32 install

popd
