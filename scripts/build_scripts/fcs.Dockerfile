ARG BASE=docker.io/dingpf/root:6.30.04-ubuntu22.04 
FROM $BASE

ARG WORK_DIR=/hep-mini-apps
ARG ROOT_INSTALL_DIR=$WORK_DIR/root/install

ARG FCS_SRC_DIR=$WORK_DIR/FCS-GPU/source
ARG FCS_BUILD_DIR=$WORK_DIR/FCS-GPU/build
ARG FCS_INSTALL_DIR=$WORK_DIR/FCS-GPU/install
ARG FCS_BRANCH=dingpf/packaging

RUN \
  cd $ROOT_INSTALL_DIR/bin && \
  . $ROOT_INSTALL_DIR/bin/thisroot.sh && \
  mkdir -p $FCS_BUILD_DIR && \
  mkdir -p $FCS_INSTALL_DIR && \
  git clone https://github.com/cgleggett/FCS-GPU.git -b ${FCS_BRANCH} $FCS_SRC_DIR && \
  cd $FCS_BUILD_DIR && \
  cmake -DCMAKE_INSTALL_PREFIX=$FCS_INSTALL_DIR \
	-DENABLE_XROOTD=Off \
	-DCMAKE_CXX_STANDARD=17 \
	-DCMAKE_CXX_EXTENSIONS=Off \
	$FCS_SRC_DIR/FastCaloSimAnalyzer   && \
  make -j 128 install && \
  rm -rf $FCS_BUILD_DIR

