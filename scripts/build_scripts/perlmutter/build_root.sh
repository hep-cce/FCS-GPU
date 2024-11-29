#!/bin/bash

# On Perlmutter
#


ROOT_VERSION=v6-30-04

module load python

WORK_DIR=$SCRATCH/hep-mini-apps

SRC_DIR=$WORK_DIR/root_src
INSTALL_DIR=$WORK_DIR/root_install
BUILD_DIR=$WORK_DIR/root_build
mkdir -p $WORK_DIR
git clone --branch $ROOT_VERSION --depth=1 https://github.com/root-project/root.git $SRC_DIR

rm -rf $BUILD_DIR
mkdir -p $BUILD_DIR
mkdir -p $INSTALL_DIR

cd $BUILD_DIR 
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
	$SRC_DIR
make -j 128 install
# took about 15 minutes with -j 128, source, build, and install dir are all on $PSCRATCH
