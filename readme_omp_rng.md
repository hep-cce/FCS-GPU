
## Build Instructions for alpha/lambda @ CSI, BNL
Change OMP_RNG path in FastCaloSimAnalyzer/FastCaloGpu/src/CMakeLists.txt
according to your location of
git clone https://github.com/GKNB/test-benchmark-OpenMP-atomic.git

```
module use /work/software/modulefiles
module load llvm-openmp-dev
source /work/atif/packages/root-6.24-gcc-9.3.0/bin/thisroot.sh
export FCS_DATAPATH=/work/atif/FastCaloSimInputs/
export OMP_TARGET_OFFLOAD=mandatory
cmake ../FastCaloSimAnalyzer -DENABLE_XROOTD=off -DENABLE_GPU=on -DENABLE_OMPGPU=on -DRNDGEN_OMP=on  -DCMAKE_CXX_COMPILER=clang++ -DCMAKE_CXX_STANDARD=14  -DCUDA_CUDART_LIBRARY=/usr/local/cuda/lib64/libcudart.so -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda/ -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc -DCMAKE_CXX_FLAGS="-DARCH_CUDA -I/usr/local/cuda/include"
```

# For AMD
cmake ../FastCaloSimAnalyzer -DENABLE_XROOTD=off -DENABLE_GPU=on -DENABLE_OMPGPU=on -DRNDGEN_OMP=on  -DCMAKE_CXX_COMPILER=clang++ -DCMAKE_CXX_STANDARD=14  -DCUDA_CUDART_LIBRARY=/usr/local/cuda/lib64/libcudart.so -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda/ -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc -DCMAKE_CXX_FLAGS="-DARCH_HIP -I/opt/rocm/include -L/opt/rocm/rocrand/lib/ -lrocrand"
