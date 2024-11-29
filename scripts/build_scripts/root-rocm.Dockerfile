ARG BASE=rocm/rocm-terminal:6.2.1
ARG REFRESHED_AT=2024-11-28
FROM $BASE

ARG DEBIAN_FRONTEND noninteractive

USER root
RUN \
    DEBIAN_FRONTEND=${DEBIAN_FRONTEND} \
    apt-get update && \
    DEBIAN_FRONTEND=${DEBIAN_FRONTEND} \
    apt-get upgrade --yes && \
        apt-get install --yes \
        hiprand \
        rocrand && \
    apt-get clean all

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
     make -j 64 install && \
     rm -rf $ROOT_BUILD_DIR
