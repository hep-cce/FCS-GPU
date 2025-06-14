#!/bin/bash

#module load python
#module load cmake/3.24.3

INSTALL_DIR=$WORK/hep-mini-apps-kokkos/kokkos_install
SRC_DIR=$WORK/hep-mini-apps-kokkos/kokkos
BUILD_DIR=$WORK/hep-mini-apps-kokkos/kokkos_build
Kokkos_BRANCH=4.5.01


mkdir -p $INSTALL_DIR
mkdir -p $BUILD_DIR
rm -rf $SRC_DIR


ml cuda/12.5
ml gcc/13.2.0

export CC=/opt/apps/gcc/13.2.0/bin/gcc
export CXX=/opt/apps/gcc/13.2.0/bin/g++
git clone https://github.com/kokkos/kokkos.git -b ${Kokkos_BRANCH} $SRC_DIR 

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
      -DKokkos_ARCH_HOPPER90=ON \
      -DBUILD_SHARED_LIBS=ON \
      ${SRC_DIR}

#Kokkos_ARCH_HOPPER90=ON

make -j 32 install

popd
