# FastCaloSim GPU Project

## Table of contents
* [BNL Build Instructios](#Build-Instructions-for-BNL)
* [Cori Build Instructios](#Build-Instructions-for-Cori)
* [Kokkos](#Kokkos)
* [Formatting](#Formatting)

## Build Instructions for BNL

load build environment with appropriate modules
```
module load cmake gcc/6.4.0 cuda/9.0 root/v6.14.08-gcc-6.4.0
```
or
```
## module load cmake gcc/8.2.0 cuda/10.1 root/6.18.02-gcc-8.2.0
module load cmake gcc/8.2.0 llvm/13.0.0 root/v6.20.04-gcc-8.2.0
```

then

```
mkdir build 
cd build
## cmake ../FastCaloSimAnalyzer -DENABLE_XROOTD=off -DENABLE_GPU=on -DINPUT_PATH="/hpcgpfs01/work/csi/cce/FastCaloSimInputs"
cmake ../FastCaloSimAnalyzer -DENABLE_XROOTD=off -DENABLE_GPU=off -DCMAKE_CXX_COMPILER=clang++ -DCMAKE_CXX_STANDARD=14 -DINPUT_PATH="/hpcgpfs01/work/csi/cce/FastCaloSimInputs"
make -j 8
```

load the runtime environment:
```
## . x86_64-slc7-gcc8-opt/setup.sh
. x86_64-slc7-clang130-opt/setup.sh
```
run the simulation

```
cd $your_work_directy
srun -A cce -p long -N1 -n1 runTFCSSimulation
```

## Build Instructions for Cori

### Building ROOT
ROOT v 6.14.08 (from tag) has been built with gcc8.3 and c++14 and installed in
`/global/cfs/cdirs/atlas/leggett/root/v6-14-08_gcc83_c14`

While FastCaloSim can be built with c++17 support, nvcc (for CUDA) can
only handle c++14


```
ROOT was built with
module load gcc/8.3.0
module load cmake/3.14.4

export CC=`which gcc`
export CXX=`which g++`

cmake -Dcxx=14 -Dcxx14=ON ../src
make -j30 VERBOSE=1 >& make.log
```

an example script is [here](scripts/cori/install_root_cxx14.sh)

### Building FastCaloSim

(code checked out in directory "src")

```
module load cuda
module load gcc/8.3.0
module load cmake/3.14.4

export ROOTDIR=/global/cfs/cdirs/atlas/leggett/root/v6-14-08_gcc83
export CMAKE_PREFIX_PATH=$ROOTDIR
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$ROOTDIR/lib
export PATH=$PATH:$ROOTDIR/bin

export CC=`which gcc`
export CXX=`which g++`

mkdir build
cd build
cmake ../src/FastCaloSimAnalyzer -DENABLE_XROOTD=off -DENABLE_GPU=on -DINPUT_PATH="/global/cfs/cdirs/atlas/leggett/data/FastCaloSimInputs" -DCMAKE_CXX_STANDARD=14
make -j 30 VERBOSE=1 >& make.log
```

to disable GPU, set option `-DENABLE_GPU=off`

### To run on GPU nodes

```
source $BUILD_DIR/x86_64-linux15-gcc8-opt/setup.sh

module load esslurm
salloc -N 1 -t 30 -c 80 --gres=gpu:8 --exclusive -C gpu -A m1759
srun -N1 -n1 runTFCSSimulation
```

## Kokkos

If Kokkos with a CUDA backend is already installed, ensure that the
environment variables `KOKKOS_ROOT` points to the install area, and that
`nvcc_wrapper` is in your `PATH`. Otherwise:

### Install Kokkos With CUDA backend
Checkout Kokkos from `git@github.com:kokkos/kokkos.git` into `$KOKKOS_SRC`

Set the env var `KOKKOS_INSTALL_DIR` to an appropriate value. CPU and
GPU architectures must be chosen. See [here](https://github.com/kokkos/kokkos/wiki/Compiling). For example, on a Haswell CPU and V100 GPU:

Build with
```
cmake ../src \
-DCMAKE_INSTALL_PREFIX=${KOKKOS_INSTALL_DIR} \
-DKokkos_ARCH_HSW=${CPU_ARCH} \
-DBUILD_SHARED_LIBS=On -DKokkos_ENABLE_CUDA_RELOCATABLE_DEVICE_CODE=Off \
-DKokkos_ENABLE_CUDA=On -DKokkos_ARCH_VOLTA72=On -DKokkos_ENABLE_CUDA_LAMBDA=On \
-DKokkos_ENABLE_SERIAL=On \
-DKokkos_CXX_STANDARD=14 \
-DCMAKE_CXX_COMPILER=${KOKKOS_SRC}/bin/nvcc_wrapper

make
make install
```

An example script to build and install Kokkos is [here](scripts/cori/install_kokkos.sh)


### Build FastCaloSim with CUDA enabled Kokkos

To build with Kokkos instead of plain nvcc, make sure you have the Kokkos
environment loaded, and that `$CXX` points to `nvcc_wrapper` from Kokkos.

Then add `-DUSE_KOKKOS=on` to the FastCaloSim cmake configuration


## Validation
### Random Numbers

Random numbers are by default generated on the GPU if `-DENABLE_GPU=On` is set
during configuration. Alternatively, to help comparison between CPU and GPU codes,
the random numbers can be generated on the CPU and transferred to the GPU. This is
enabled by setting the cmake configuration parameter `-DRNDGEN_CPU=On`.


### Checking Output

The number of hit cells and counts can be displayed by setting the environment
variable `FCS_DUMP_HITCOUNT=1` at run time. This will result in an output like:
```
 HitCellCount:  12 /  12   nhit:   55   
 HitCellCount:  48 /  60   nhit: 1220  *
 HitCellCount:  76 / 136   nhit: 4944  *
 HitCellCount:   6 / 142   nhit:   10   
 HitCellCount:   1 / 143   nhit:    1
```
Lines marked with an asterisk were executed on the GPU.


## Formatting
The FastCaloSimAnalyzer code has been formatted with:

```
find FastCaloSimAnalyzer -type f -iname \*.h -o -iname \*.cpp -o -iname \*.cxx -o -iname \*.cu | xargs clang-format-9 -i -style=file
```
