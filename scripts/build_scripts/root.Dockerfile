ARG CUDA_VERSION=12.2.2
ARG CUDA_BASE=docker.io/nvidia/cuda:$CUDA_VERSION-devel-ubuntu22.04
ARG NVHPC_CUDA_VERSION=23.9-devel-cuda12.2
ARG NVHPC_BASE=nvcr.io/nvidia/nvhpc:${NVHPC_CUDA_VERSION}-ubuntu22.04
ARG UBUNTU_BASE=docker.io/library/ubuntu:22.04

ARG BASE=$CUDA_BASE
FROM $BASE

ARG DEBIAN_FRONTEND noninteractive

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
ARG ROOT_SRC_DIR=$WORK_DIR/root/source
ARG ROOT_INSTALL_DIR=$WORK_DIR/root/install
ARG ROOT_BUILD_DIR=$WORK_DIR/build
RUN \
    mkdir -p $ROOT_BUILD_DIR && \
    git clone --branch $ROOT_VERSION --depth=1 https://github.com/root-project/root.git $ROOT_SRC_DIR && \
    mkdir -p $ROOT_INSTALL_DIR && \
    cd $ROOT_BUILD_DIR && \
    cmake -DCMAKE_INSTALL_PREFIX=$ROOT_INSTALL_DIR \
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
          $ROOT_SRC_DIR && \
     make -j 128 install && \
     rm -rf $ROOT_BUILD_DIR
