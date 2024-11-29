ARG BASE=registry.nersc.gov/m2845/root:6.30.04-rocm-terminal6.2.1
ARG REFRESHED_AT=2024-11-28
FROM $BASE

USER root

ARG DEBIAN_FRONTEND noninteractive
RUN \
    DEBIAN_FRONTEND=${DEBIAN_FRONTEND} \
    apt-get update && \
    DEBIAN_FRONTEND=${DEBIAN_FRONTEND} \
    apt-get upgrade --yes && \
        apt-get install --yes \
        wget && \
    apt-get clean all

RUN \
    wget https://github.com/Kitware/CMake/releases/download/v3.31.1/cmake-3.31.1-linux-x86_64.sh &&\
    chmod +x cmake-3.31.1-linux-x86_64.sh  && \
    mkdir -p /opt/cmake && \
    ./cmake-3.31.1-linux-x86_64.sh --prefix=/opt/cmake --skip-license --exclude-subdir && \
    rm -f ./cmake-3.31.1-linux-x86_64.sh 

ENV PATH=/opt/cmake/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/opt/rocm/bin

ARG WORK_DIR=/hep-mini-apps
ARG FCS_SRC_DIR=$WORK_DIR/FCS-GPU/source
ARG FCS_BUILD_DIR=$WORK_DIR/FCS-GPU/build
ARG FCS_INSTALL_DIR=$WORK_DIR/FCS-GPU/install
ARG FCS_BRANCH=dingpf/packaging
ARG ROOT_INSTALL_DIR=$WORK_DIR/root/install

RUN \
  cd $ROOT_INSTALL_DIR/bin && \
  . $ROOT_INSTALL_DIR/bin/thisroot.sh && \
  mkdir -p $FCS_BUILD_DIR && \
  mkdir -p $FCS_INSTALL_DIR && \
  git clone https://github.com/hep-cce/FCS-GPU.git -b ${FCS_BRANCH} $FCS_SRC_DIR && \
  cd $FCS_BUILD_DIR && \
  cmake -DCMAKE_INSTALL_PREFIX=$FCS_INSTALL_DIR \
        -DUSE_HIP=on \
        -DCMAKE_CXX_COMPILER=hipcc \
        -DCMAKE_CXX_STANDARD=17 \
        -DCMAKE_CXX_EXTENSIONS=Off \
        -DENABLE_GPU=on \
        $FCS_SRC_DIR/FastCaloSimAnalyzer   && \
  make -j 128 install && \
  cd $WORK_DIR && \
  rm -rf $FCS_BUILD_DIR

