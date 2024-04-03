ARG NVHPC_CUDA_VERSION=23.9-devel-cuda12.2
FROM nvcr.io/nvidia/nvhpc:${NVHPC_CUDA_VERSION}-ubuntu22.04

ENV LANG=C.UTF-8

WORKDIR /opt

RUN \
    apt-get update && \
    apt-get upgrade --yes && \
        apt-get install --yes \
        build-essential \
        cmake \
        wget \
        vim \
        python3 \
        git && \
    apt-get clean all

ENV CUDA_PATH /usr/local/cuda/

ARG ROOT_VERSION=v6-30-04
ARG WORK_DIR=/hep-mini-apps
ARG SRC_DIR=$WORK_DIR/root_src
ARG INSTALL_DIR=$WORK_DIR/root_install
ARG BUILD_DIR=$WORK_DIR/root_build
RUN \
    mkdir -p $WORK_DIR && \
    git clone --branch $ROOT_VERSION --depth=1 https://github.com/root-project/root.git $SRC_DIR && \
    mkdir -p $BUILD_DIR  && \
    mkdir -p $INSTALL_DIR && \
    cd $BUILD_DIR && \
    cmake -DCMAKE_INSTALL_PREFIX=$INSTALL_DIR \
          -DCMAKE_CXX_FLAGS=-std=c++17 \
          -Dx11=OFF -Dtbb=OFF \
          -Dopengl=OFF -Dgviz=OFF \
          -Dimt=OFF -Ddavix=OFF \
          -Dvdt=OFF -Dxrootd=OFF \
          -Dwebgui=OFF -Dsqlite=OFF \
          -Dssl=OFF -Dmysql=OFF \
          -Doracle=OFF -Dpgsql=OFF \
          -Ddavix=OFF -Dgfal=OFF \
          -Dimt=OFF \
          -DCMAKE_CXX_STANDARD=17 \
          -DCMAKE_CXX_EXTENSIONS=Off \
          $SRC_DIR && \
     make -j 128 install && \
     rm -rf $BUILD_DIR 

ARG ROOT_DIR=$WORK_DIR/root_install
ARG FCS_SRC_DIR=$WORK_DIR/FCS-GPU_src
ARG FCS_BUILD_DIR=$WORK_DIR/FCS-GPU_gpu_build
ARG FCS_INSTALL_DIR=$WORK_DIR/FCS-GPU_gpu_install

RUN \
  cd $ROOT_DIR/bin && \
 . $ROOT_DIR/bin/thisroot.sh && \
 mkdir -p $FCS_BUILD_DIR && \
 mkdir -p $FCS_INSTALL_DIR && \
 git clone https://github.com/cgleggett/FCS-GPU.git -b dingpf/packaging $FCS_SRC_DIR && \
 cd $FCS_BUILD_DIR && \
 cmake -DCMAKE_INSTALL_PREFIX=$FCS_INSTALL_DIR \
       -DENABLE_XROOTD=Off \
       -DCMAKE_CXX_STANDARD=17 \
       -DCMAKE_CXX_EXTENSIONS=Off \
      	-DENABLE_GPU=on \
       -DCMAKE_CUDA_ARCHITECTURES=80 \
       $FCS_SRC_DIR/FastCaloSimAnalyzer   && \
 make -j 128 install
