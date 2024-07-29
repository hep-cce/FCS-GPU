ARG BASE=registry.nersc.gov/m2845/root:6.30.04-cuda12.2.2-devel-ubuntu22.04
FROM $BASE

ARG DEBIAN_FRONTEND noninteractive

RUN \
    DEBIAN_FRONTEND=${DEBIAN_FRONTEND} \
    apt-get update && \
    DEBIAN_FRONTEND=${DEBIAN_FRONTEND} \
    apt-get upgrade --yes && \
        apt-get install --yes \
        build-essential \
        cmake \
        wget \
        vim \
        python3 \
        git && \
    apt-get clean all

RUN \
    wget https://repo.radeon.com/rocm/rocm.gpg.key -O - | gpg --dearmor | tee /etc/apt/keyrings/rocm.gpg > /dev/null; \
    echo "deb [arch=amd64 signed-by=/etc/apt/keyrings/rocm.gpg] https://repo.radeon.com/rocm/apt/6.1.1 jammy main"     | tee --append /etc/apt/sources.list.d/rocm.list; \
    DEBIAN_FRONTEND=${DEBIAN_FRONTEND}  apt update && \
    apt install --yes hipcc

ENV CUDA_PATH /usr/local/cuda/

    #mkdir -p /opt &&  cd /opt &&  export ROCMBRANCH=rocm-6.1.x && \

ARG ROCM_BRANCH=rocm-6.1.x

RUN \
    mkdir -p /opt &&  cd /opt && \
    git clone -b ${ROCM_BRANCH} https://github.com/ROCm/clr.git && \
    git clone -b ${ROCM_BRANCH} https://github.com/ROCm/hip.git && \
    git clone -b ${ROCM_BRANCH} https://github.com/ROCm/hipother.git && \
    export CLR_DIR="$(readlink -f clr)" && \
    export HIP_DIR="$(readlink -f hip)" && \
    export HIP_OTHER="$(readlink -f hipother)" && \
    cd $CLR_DIR && \
    mkdir -p build && cd build && \
    export HIP_PLATFORM=nvidia && \
    cmake -DHIP_COMMON_DIR=$HIP_DIR -DHIP_PLATFORM=nvidia -DCMAKE_INSTALL_PREFIX=$PWD/install -DHIP_CATCH_TEST=0 -DCLR_BUILD_HIP=ON -DCLR_BUILD_OCL=OFF -DHIPNV_DIR=$HIP_OTHER/hipnv .. && \
    make -j 10  && \
    make install

ENV HIP_PLATFORM=nvidia


ARG WORK_DIR=/hep-mini-apps
ARG ROOT_INSTALL_DIR=$WORK_DIR/root/install

ARG FCS_SRC_DIR=$WORK_DIR/FCS-GPU/source
ARG FCS_BUILD_DIR=$WORK_DIR/FCS-GPU/build
ARG FCS_INSTALL_DIR=$WORK_DIR/FCS-GPU/install
ARG FCS_BRANCH=dingpf/packaging

RUN \
  export PATH=/opt/clr/build/install/bin:$PATH && \
  cd $ROOT_INSTALL_DIR/bin && \
  . $ROOT_INSTALL_DIR/bin/thisroot.sh && \
  mkdir -p $FCS_BUILD_DIR && \
  mkdir -p $FCS_INSTALL_DIR && \
  git clone https://github.com/hep-cce/FCS-GPU.git -b ${FCS_BRANCH} $FCS_SRC_DIR && \
  cd $FCS_BUILD_DIR && \
# currently not working due to missing patches of fcs
#   cmake -DCMAKE_INSTALL_PREFIX=$FCS_INSTALL_DIR \
# 	-DENABLE_XROOTD=Off \
# 	-DCMAKE_CXX_STANDARD=17 \
# 	-DCMAKE_CXX_EXTENSIONS=Off \
#        	-DENABLE_GPU=on \
#         -DCMAKE_CXX_COMPILER=hipcc \
#         -DCMAKE_CUDA_ARCHITECTURES=80 \
# 	$FCS_SRC_DIR/FastCaloSimAnalyzer   && \
#   make -j 128 install && \
  rm -rf $FCS_BUILD_DIR

	
