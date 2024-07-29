ARG BASE=registry.nersc.gov/m2845/root:6.30.04-cuda12.2.2-devel-ubuntu22.04
FROM $BASE

# ARG BASE=docker.io/dingpf/root:6.30.04-cuda12.2.2-ubuntu22.04 
#ARG BASE=docker.io/dingpf/root:6.30.04-nvhpc23.9-cuda12.2-ubuntu22.04

ARG WORK_DIR=/hep-mini-apps
ARG ROOT_INSTALL_DIR=$WORK_DIR/root/install

ARG Kokkos_SRC_DIR=$WORK_DIR/Kokkos/source
ARG Kokkos_BUILD_DIR=$WORK_DIR/Kokkos/build
ARG Kokkos_INSTALL_DIR=$WORK_DIR/Kokkos/install
ARG Kokkos_BRANCH=4.2.01
RUN \
  mkdir -p $Kokkos_BUILD_DIR && \
  mkdir -p $Kokkos_INSTALL_DIR && \
  git clone https://github.com/kokkos/kokkos.git -b ${Kokkos_BRANCH} $Kokkos_SRC_DIR && \
  cd $Kokkos_BUILD_DIR && \
  cmake -DCMAKE_INSTALL_PREFIX=${Kokkos_INSTALL_DIR} \
      -DCMAKE_CXX_COMPILER=${Kokkos_SRC_DIR}/bin/nvcc_wrapper \
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
      ${Kokkos_SRC_DIR} && \
  make -j 32 install && \
  rm -rf $Kokkos_BUILD_DIR

ARG FCS_SRC_DIR=$WORK_DIR/FCS-GPU/source
ARG FCS_BUILD_DIR=$WORK_DIR/FCS-GPU/build
ARG FCS_INSTALL_DIR=$WORK_DIR/FCS-GPU/install
ARG FCS_BRANCH=dingpf/packaging
ENV CMAKE_PREFIX_PATH="${Kokkos_INSTALL_DIR}:${CMAKE_PREFIX_PATH}"
RUN \
  cd $ROOT_INSTALL_DIR/bin && \
  . $ROOT_INSTALL_DIR/bin/thisroot.sh && \
  mkdir -p $FCS_BUILD_DIR && \
  mkdir -p $FCS_INSTALL_DIR && \
  git clone https://github.com/hep-cce/FCS-GPU.git -b ${FCS_BRANCH} $FCS_SRC_DIR && \
  cd $FCS_BUILD_DIR && \
  cmake -DCMAKE_INSTALL_PREFIX=$FCS_INSTALL_DIR \
	  -DENABLE_XROOTD=Off \
	  -DCMAKE_CXX_STANDARD=17 \
	  -DCMAKE_CXX_EXTENSIONS=Off \
    -DENABLE_GPU=on \
	  -DCMAKE_CUDA_ARCHITECTURES=80 \
    -DUSE_KOKKOS=ON \
    -DCMAKE_CXX_COMPILER=$Kokkos_INSTALL_DIR/bin/nvcc_wrapper \
  	$FCS_SRC_DIR/FastCaloSimAnalyzer   && \
  make -j 16 install && \
  rm -rf $FCS_BUILD_DIR