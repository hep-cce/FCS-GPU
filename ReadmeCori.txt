Instructions to run on cori.nersc.gov
=====================================

ROOT v 6.14.08 (from tag) has been built with gcc8.3 and c++17 and installed in
/global/project/projectdirs/atlas/leggett/root/v6-14-08_gcc83


ROOT was built with
module load gcc/8.3.0
module load cmake/3.14.4

export CC=`which gcc`
export CXX=`which g++`

cmake -Dcxx=17 -Dcxx17=ON ../src
make -j30 VERBOSE=1 >& make.log


To build FastCaloSim
--------------------

(code checked out in directory "src")

module load cuda
module load gcc/8.3.0
module load cmake/3.14.4

export ROOTDIR=/global/project/projectdirs/atlas/leggett/root/v6-14-08_gcc83
export CMAKE_PREFIX_PATH=$ROOTDIR
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$ROOTDIR/lib
export PATH=$PATH:$ROOTDIR/bin

export CC=`which gcc`
export CXX=`which g++`

mkdir build
cd build
cmake ../src/FastCaloSimAnalyzer -DENABLE_XROOTD=off -DENABLE_GPU=on -DINPUT_PATH="/global/project/projectdirs/atlas/leggett/data/FastCaloSimInputs" -DCMAKE_CXX_STANDARD=17
make -j 30 VERBOSE=1 >& make.log


to disable GPU, set option -DENABLE_GPU=off


To run on GPU nodes
-------------------

source $BUILD_DIR/x86_64-linux15-gcc8-opt/setup.sh

module load esslurm
salloc -N 1 -t 30 -c 80 --gres=gpu:8 --exclusive -C gpu -A m1759
srun -N1 -n1 runTFCSSimulation
