# CPU           Exalearn5

# CUDA  -----------------        
## Nvidia  --------------
### CURAND      Exalearn5
### CPURNG      Exalearn5

# OpenMP  ---------------
## Nvidia  --------------
### CURAND      Exalearn5
### CPURNG      Exalearn5
### OMPRNG      ---------
#### ARCH_CUDA  Exalearn5
#### RANDOM123  Exalearn5
## AMD  -----------------
### ROCRAND     Exalearn4 
### CPURNG      Exalearn4
### OMPRNG      ---------
#### ARCH_HIP   Exalearn4
#### RANDOM123  Exalearn4
## Multicore CPU --------
### CPURNG      Exalearn4
### OMPRNG      ---------
#### RANDOM123  Exalearn5

# HIP  ------------------
## Nvidia  --------------
### CURAND      xxxxxxxxx
### CPURNG      Perlmutter
## AMD  -----------------
### HIPRAND     Exalearn4
### CPURNG      Exalearn4

# STDPAR  ---------------
## Nvidia  --------------
### CURAND      Exalearn5
### CPURNG      Exalearn5
## Multicore  -----------
### CPURNG      Exalearn5
## CPU  -----------------
### CPURNG      Exalearn5

# Alpaka  ---------------
## Nvidia CUDA  ---------
### CURAND      Exalearn5
### CPURNG      Exalearn5
## AMD HIP  -------------
### HIPRAND
### CPURNG

# Kokkos  ---------------
## Nvidia  --------------
### CURAND      Exalearn5
### CPURNG      Exalearn5

# Edit this to exalearn4 or 5 accordingly
system="exalearn5"

rm -rf build-exalearn4-*
rm -rf build-exalearn5-*

if [ "$system" = "exalearn4" ]; then
  source /global/home/users/fmohammad/packages/root-clang15/bin/thisroot.sh
  export FCS_DATAPATH=/global/home/users/cgleggett/data/FastCaloSimInputs
  module use /global/home/users/fmohammad/modulefiles/
  module load clang-18.0.0-gcc-8.5.0-omp-amdgcn
fi

if [ "$system" = "exalearn5" ]; then
  source /global/home/users/fmohammad/packages/root-clang15/bin/thisroot.sh
  export FCS_DATAPATH=/global/home/users/cgleggett/data/FastCaloSimInputs
  module use /global/home/users/fmohammad/modulefiles/
  #module load clang-15.0.6-gcc-8.5.0-omp-nvptx
  module load clang-17.0.0-gcc-8.5.0-omp-nvptx
fi

# # # # # # # # # # # # # #

# CPU
if [ "$system" = "exalearn5" ]; then
  echo "x-x-x-x-x CPU BUILD x-x-x-x-x"
  mkdir -p build-exalearn5-cpu
  cd build-exalearn5-cpu
  cmake ../FastCaloSimAnalyzer -DENABLE_XROOTD=off -DENABLE_GPU=off -DCMAKE_CXX_STANDARD=17
  make -j16
  echo "x-x-x-x-x CPU BUILD DONE! x-x-x-x-x"
  cd ..
fi

# # # # # # # # # # # # # #

# CUDA
## Nvidia
### CURAND
if [ "$system" = "exalearn5" ]; then
  echo "x-x-x-x-x CUDA CURAND BUILD x-x-x-x-x"
  module load cuda/11.5
  mkdir -p build-exalearn5-cuda-curand
  cd build-exalearn5-cuda-curand
  cmake ../FastCaloSimAnalyzer -DENABLE_XROOTD=off -DENABLE_GPU=on -DRNDGEN_CPU=Off -DCMAKE_CXX_COMPILER=clang++ -DCMAKE_CXX_STANDARD=17 -DCMAKE_CUDA_ARCHITECTURES=80
  make -j16
  echo "x-x-x-x-x CUDA CURAND BUILD DONE! x-x-x-x-x"
  cd ..
fi
### CPURNG
if [ "$system" = "exalearn5" ]; then
  echo "x-x-x-x-x CUDA CPURNG BUILD x-x-x-x-x"
  mkdir -p build-exalearn5-cuda-cpurng
  cd build-exalearn5-cuda-cpurng
  cmake ../FastCaloSimAnalyzer -DENABLE_XROOTD=off -DENABLE_GPU=on -DRNDGEN_CPU=On -DCMAKE_CXX_COMPILER=clang++ -DCMAKE_CXX_STANDARD=17 -DCMAKE_CUDA_ARCHITECTURES=80
  make -j16
  echo "x-x-x-x-x CUDA CPURNG BUILD DONE! x-x-x-x-x"
  cd ..
fi

# # # # # # # # # # # # # #

# OpenMP
## Nvidia
### CURAND
if [ "$system" = "exalearn5" ]; then
  echo "x-x-x-x-x OpenMP Nvidia CURAND BUILD x-x-x-x-x"
  export OMP_TARGET_OFFLOAD=mandatory
  mkdir -p build-exalearn5-openmp-nv-curand
  cd build-exalearn5-openmp-nv-curand
  cmake ../FastCaloSimAnalyzer -DENABLE_XROOTD=off -DENABLE_GPU=on -DENABLE_OMPGPU=on -DRNDGEN_CPU=Off  -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ -DCMAKE_CXX_STANDARD=17 -DCMAKE_CXX_FLAGS="--offload-arch=sm_80"
  make -j16
  echo "x-x-x-x-x OpenMP Nvidia CURAND BUILD DONE! x-x-x-x-x"
  cd ..
fi
### CPURNG
if [ "$system" = "exalearn5" ]; then
  echo "x-x-x-x-x OpenMP Nvidia CPURNG BUILD x-x-x-x-x"
  export OMP_TARGET_OFFLOAD=mandatory
  mkdir -p build-exalearn5-openmp-nv-cpurng
  cd build-exalearn5-openmp-nv-cpurng
  cmake ../FastCaloSimAnalyzer -DENABLE_XROOTD=off -DENABLE_GPU=on -DENABLE_OMPGPU=on -DRNDGEN_CPU=On  -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ -DCMAKE_CXX_STANDARD=17 -DCMAKE_CXX_FLAGS="--offload-arch=sm_80"
  make -j16
  echo "x-x-x-x-x OpenMP Nvidia CPURNG BUILD DONE x-x-x-x-x"
  cd ..
fi
### Portable OMP RNG
#### ARCH_CUDA
if [ "$system" = "exalearn5" ]; then
  echo "x-x-x-x-x OpenMP Nvidia PortableOMPRNG ARCH_CUDA BUILD x-x-x-x-x"
  export OMP_TARGET_OFFLOAD=mandatory
  mkdir -p build-exalearn5-openmp-nv-omprng-archcuda
  cd build-exalearn5-openmp-nv-omprng-archcuda
cmake ../FastCaloSimAnalyzer -DENABLE_XROOTD=off -DENABLE_GPU=on -DENABLE_OMPGPU=on -DRNDGEN_CPU=Off -DRNDGEN_OMP=On  -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ -DCMAKE_CXX_STANDARD=14 -DCMAKE_CXX_FLAGS="--offload-arch=sm_80" -DARCH_CUDA=on
  make -j16
  echo "x-x-x-x-x OpenMP Nvidia PortableOMPRNG ARCH_CUDA DONE x-x-x-x-x"
  cd ..
fi
#### USE_RANDOM123
if [ "$system" = "exalearn5" ]; then
  echo "x-x-x-x-x OpenMP Nvidia PortableOMPRNG USE_RANDOM123 BUILD x-x-x-x-x"
  export OMP_TARGET_OFFLOAD=mandatory
  mkdir -p build-exalearn5-openmp-nv-omprng-random123
  cd build-exalearn5-openmp-nv-omprng-random123
cmake ../FastCaloSimAnalyzer -DENABLE_XROOTD=off -DENABLE_GPU=on -DENABLE_OMPGPU=on -DRNDGEN_CPU=Off -DRNDGEN_OMP=On  -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ -DCMAKE_CXX_STANDARD=14 -DCMAKE_CXX_FLAGS="--offload-arch=sm_80" -DUSE_RANDOM123=on
  make -j16
  echo "x-x-x-x-x OpenMP Nvidia PortableOMPRNG USE_RANDOM123 DONE x-x-x-x-x"
  cd ..
fi

# OpenMP
## AMD
### ROCRAND
if [ "$system" = "exalearn4" ]; then
  echo "x-x-x-x-x OpenMP AMD CPURNG BUILD x-x-x-x-x"
  export OMP_TARGET_OFFLOAD=mandatory
  export ROCM_PATH=/opt/rocm/
  mkdir -p build-exalearn4-openmp-amd-rocrand
  cd build-exalearn4-openmp-amd-rocrand
  cmake ../FastCaloSimAnalyzer -DENABLE_XROOTD=off -DENABLE_GPU=on -DENABLE_OMPGPU=on -DRNDGEN_CPU=Off  -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ -DCMAKE_CXX_STANDARD=17 -DCMAKE_CXX_FLAGS="--offload-arch=gfx908"
  make -j32
  echo "x-x-x-x-x OpenMP AMD CPURNG BUILD DONE x-x-x-x-x"
  cd ..
fi
### CPURNG
if [ "$system" = "exalearn4" ]; then
  echo "x-x-x-x-x OpenMP AMD CPURNG BUILD x-x-x-x-x"
  export OMP_TARGET_OFFLOAD=mandatory
  mkdir -p build-exalearn4-openmp-amd-cpurng
  cd build-exalearn4-openmp-amd-cpurng
  cmake ../FastCaloSimAnalyzer -DENABLE_XROOTD=off -DENABLE_GPU=on -DENABLE_OMPGPU=on -DRNDGEN_CPU=On  -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ -DCMAKE_CXX_STANDARD=17 -DCMAKE_CXX_FLAGS="--offload-arch=gfx908"
  make -j32
  echo "x-x-x-x-x OpenMP AMD CPURNG BUILD DONE x-x-x-x-x"
  cd ..
fi
### Portable OMP RNG
#### ARCH_HIP
if [ "$system" = "exalearn4" ]; then
  echo "x-x-x-x-x OpenMP AMD PortableOMPRNG ARCH_HIP BUILD x-x-x-x-x"
  export OMP_TARGET_OFFLOAD=mandatory
  mkdir -p build-exalearn5-openmp-amd-omprng-archhip
  cd build-exalearn5-openmp-amd-omprng-archhip
cmake ../FastCaloSimAnalyzer -DENABLE_XROOTD=off -DENABLE_GPU=on -DENABLE_OMPGPU=on -DRNDGEN_CPU=Off -DRNDGEN_OMP=On  -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ -DCMAKE_CXX_STANDARD=14 -DCMAKE_CXX_FLAGS="--offload-arch=gfx908" -DARCH_HIP=on
  make -j16
  echo "x-x-x-x-x OpenMP AMD PortableOMPRNG ARCH_HIP DONE x-x-x-x-x"
  cd ..
fi
#### USE_RANDOM123
if [ "$system" = "exalearn4" ]; then
  echo "x-x-x-x-x OpenMP AMD PortableOMPRNG USE_RANDOM123 BUILD x-x-x-x-x"
  export OMP_TARGET_OFFLOAD=mandatory
  mkdir -p build-exalearn5-openmp-nv-omprng-random123
  cd build-exalearn5-openmp-nv-omprng-random123
cmake ../FastCaloSimAnalyzer -DENABLE_XROOTD=off -DENABLE_GPU=on -DENABLE_OMPGPU=on -DRNDGEN_CPU=Off -DRNDGEN_OMP=On  -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ -DCMAKE_CXX_STANDARD=14 -DCMAKE_CXX_FLAGS="--offload-arch=gfx908" -DUSE_RANDOM123=on
  make -j16
  echo "x-x-x-x-x OpenMP AMD PortableOMPRNG USE_RANDOM123 DONE x-x-x-x-x"
  cd ..
fi


## Multicore CPU
### CPURNG
if [ "$system" = "exalearn4" ]; then
  echo "x-x-x-x-x OpenMP MULTICORE CPU CPURNG BUILD x-x-x-x-x"
  export OMP_TARGET_OFFLOAD=disabled
  mkdir -p build-exalearn4-openmp-multicorecpu-cpurng
  cd build-exalearn4-openmp-multicorecpu-cpurng
  cmake ../FastCaloSimAnalyzer -DENABLE_XROOTD=off -DENABLE_GPU=on -DENABLE_OMPGPU=on -DRNDGEN_CPU=On  -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ -DCMAKE_CXX_STANDARD=17 -DCMAKE_CXX_FLAGS="--offload-arch=gfx908" 
  make -j32
  echo "x-x-x-x-x OpenMP MULTICORE CPU CPURNG BUILD DONE x-x-x-x-x"
  cd ..
fi
### Portable-OMP-RNG
#### USE_RANDOM123
if [ "$system" = "exalearn5" ]; then
  echo "x-x-x-x-x OpenMP Nvidia PortableOMPRNG USE_RANDOM123 BUILD x-x-x-x-x"
  export OMP_TARGET_OFFLOAD=disabled
  mkdir -p build-exalearn5-openmp-multicorecpu-omprng-random123
  cd build-exalearn5-openmp-multicorecpu-omprng-random123
cmake ../FastCaloSimAnalyzer -DENABLE_XROOTD=off -DENABLE_GPU=on -DENABLE_OMPGPU=on -DRNDGEN_CPU=Off -DRNDGEN_OMP=On  -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ -DCMAKE_CXX_STANDARD=14 -DCMAKE_CXX_FLAGS="--offload-arch=sm_80" -DUSE_RANDOM123=on
  make -j16
  echo "x-x-x-x-x OpenMP Nvidia PortableOMPRNG USE_RANDOM123 DONE x-x-x-x-x"
  cd ..
fi


# # # # # # # # # # # # # #

# HIP
## Nvidia
### CURAND
### CPURNG

## HIP
## AMD
### HIPRAND
if [ "$system" = "exalearn4" ]; then
  echo "x-x-x-x-x HIP HIPRAND BUILD x-x-x-x-x"
  mkdir -p build-exalearn4-hip-amd-hiprand
  cd build-exalearn4-hip-amd-hiprand
  cmake ../FastCaloSimAnalyzer -DENABLE_XROOTD=Off -DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILER=hipcc -DCMAKE_CXX_STANDARD=17 -DCMAKE_CXX_EXTENSIONS=Off -DENABLE_GPU=on -DUSE_HIP=on -DHIP_TARGET=AMD -DRNDGEN_CPU=Off
  make -j32
  echo "x-x-x-x-x HIP HIPRAND BUILD DONE! x-x-x-x-x"
  cd ..
fi
### CPURNG
if [ "$system" = "exalearn4" ]; then
  echo "x-x-x-x-x HIP CPURNG BUILD x-x-x-x-x"
  mkdir -p build-exalearn4-hip-amd-cpurng
  cd build-exalearn4-hip-amd-cpurng
  cmake ../FastCaloSimAnalyzer -DENABLE_XROOTD=Off -DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILER=hipcc -DCMAKE_CXX_STANDARD=17 -DCMAKE_CXX_EXTENSIONS=Off -DENABLE_GPU=on -DUSE_HIP=on -DHIP_TARGET=AMD -DRNDGEN_CPU=On
  make -j32
  echo "x-x-x-x-x HIP CPURNG BUILD DONE! x-x-x-x-x"
  cd ..
fi

# # # # # # # # # # # # # #

# STDPAR
## Nvidia
### CURAND
if [ "$system" = "exalearn5" ]; then
  echo "x-x-x-x-x std::par Nvidia CURAND BUILD x-x-x-x-x"
  module purge
  root/6.24.06-gcc85-c17
  module load nvhpc/22.9
  module load cuda/11.5
  mkdir -p build-exalearn5-stdpar-nv-curand
  cd build-exalearn5-stdpar-nv-curand
  cmake ../FastCaloSimAnalyzer -DENABLE_XROOTD=Off -DCMAKE_CXX_STANDARD=17 -DCMAKE_CXX_EXTENSIONS=Off -DCMAKE_CXX_COMPILER=/global/home/users/fmohammad/FCS-GPU//scripts/nvc++_p -DENABLE_GPU=on -DUSE_STDPAR=ON -DSTDPAR_TARGET=gpu -DCMAKE_CUDA_ARCHITECTURES=80 -DRNDGEN_CPU=Off
  make -j16
  echo "x-x-x-x-x std::par Nvidia CURAND BUILD DONE x-x-x-x-x"
  cd ..
fi

### CPURNG
if [ "$system" = "exalearn5" ]; then
  echo "x-x-x-x-x std::par Nvidia CPURNG BUILD x-x-x-x-x"
  mkdir -p build-exalearn5-stdpar-nv-cpurng
  cd build-exalearn5-stdpar-nv-cpurng
  cmake ../FastCaloSimAnalyzer -DENABLE_XROOTD=Off -DCMAKE_CXX_STANDARD=17 -DCMAKE_CXX_EXTENSIONS=Off -DCMAKE_CXX_COMPILER=/global/home/users/fmohammad/FCS-GPU//scripts/nvc++_p -DENABLE_GPU=on -DUSE_STDPAR=ON -DSTDPAR_TARGET=gpu -DCMAKE_CUDA_ARCHITECTURES=80 -DRNDGEN_CPU=On
  make -j16
  echo "x-x-x-x-x std::par Nvidia CPURNG BUILD DONE! x-x-x-x-x"
  cd ..
fi

## Multicore
### CPURNG
if [ "$system" = "exalearn5" ]; then
  echo "x-x-x-x-x std::par Multicore CPURNG BUILD x-x-x-x-x"
  mkdir -p build-exalearn5-stdpar-multicore
  cd build-exalearn5-stdpar-multicore
  cmake ../FastCaloSimAnalyzer -DENABLE_XROOTD=Off -DCMAKE_CXX_STANDARD=17 -DCMAKE_CXX_EXTENSIONS=Off -DCMAKE_CXX_COMPILER=/global/home/users/fmohammad/FCS-GPU//scripts/nvc++_p -DENABLE_GPU=on -DUSE_STDPAR=ON -DSTDPAR_TARGET=multicore -DCMAKE_CUDA_ARCHITECTURES=80 -DRNDGEN_CPU=On
  make -j16
  echo "x-x-x-x-x std::par Multicore CPURNG BUILD DONE! x-x-x-x-x"
  cd ..
fi

## CPU
### CPURNG
if [ "$system" = "exalearn5" ]; then
  echo "x-x-x-x-x std::par CPU CPURNG BUILD x-x-x-x-x"
  mkdir -p build-exalearn5-stdpar-cpu
  cd build-exalearn5-stdpar-cpu
  cmake ../FastCaloSimAnalyzer -DENABLE_XROOTD=Off -DCMAKE_CXX_STANDARD=17 -DCMAKE_CXX_EXTENSIONS=Off -DCMAKE_CXX_COMPILER=/global/home/users/fmohammad/FCS-GPU//scripts/nvc++_p -DENABLE_GPU=on -DUSE_STDPAR=ON -DSTDPAR_TARGET=cpu -DCMAKE_CUDA_ARCHITECTURES=80 -DRNDGEN_CPU=On
  make -j16
  echo "x-x-x-x-x std::par CPU CPURNG BUILD DONE! x-x-x-x-x"
  cd ..
fi

# # # # # # # # # # # # # #

# Alpaka
## Nvidia CUDA
### CURAND
if [ "$system" = "exalearn5" ]; then
  echo "x-x-x-x-x Alpaka Nvidia CURAND BUILD x-x-x-x-x"
  module purge
  root/6.24.06-gcc85-c17
  module load alpaka/0.9.0
  module load cuda/11.5
  mkdir -p build-exalearn5-alpaka-nv-curand
  cd build-exalearn5-alpaka-nv-curand
  cmake ../FastCaloSimAnalyzer -DENABLE_XROOTD=off -DENABLE_GPU=on -DCMAKE_CXX_STANDARD=17 -DUSE_ALPAKA=on -Dalpaka_ROOT=/opt/alpaka/0.9.0/ -Dalpaka_ACC_GPU_CUDA_ENABLE=ON -Dalpaka_ACC_GPU_CUDA_ONLY_MODE=ON -DRNDGEN_CPU=Off -DCMAKE_CUDA_ARCHITECTURES=80
  make -j16
  echo "x-x-x-x-x Alpaka Nvidia CURAND BUILD DONE! x-x-x-x-x"
  cd ..
fi

### CPURNG
if [ "$system" = "exalearn5" ]; then
  echo "x-x-x-x-x Alpaka Nvidia CURAND BUILD x-x-x-x-x"
  mkdir -p build-exalearn5-alpaka-nv-cpurng
  cd build-exalearn5-alpaka-nv-cpurng
  cmake ../FastCaloSimAnalyzer -DENABLE_XROOTD=off -DENABLE_GPU=on -DCMAKE_CXX_STANDARD=17 -DUSE_ALPAKA=on -Dalpaka_ROOT=/opt/alpaka/0.9.0/ -Dalpaka_ACC_GPU_CUDA_ENABLE=ON -Dalpaka_ACC_GPU_CUDA_ONLY_MODE=ON -DRNDGEN_CPU=On -DCMAKE_CUDA_ARCHITECTURES=80
  make -j16
  echo "x-x-x-x-x Alpaka Nvidia CURAND BUILD DONE! x-x-x-x-x"
  cd ..
fi

## AMD HIP
### HIPRAND
### CPURNG


# # # # # # # # # # # # # #

# Kokkos
## Nvidia
### CURAND
if [ "$system" = "exalearn5" ]; then
  echo "x-x-x-x-x Kokkos Nvidia CURAND BUILD x-x-x-x-x"
  module purge
  root/6.24.06-gcc85-c17
  module load kokkos/4.1-cuda11.5-shlib
  mkdir -p build-exalearn5-kokkos-nv-curand
  cd build-exalearn5-kokkos-nv-curand
  cmake ../FastCaloSimAnalyzer -DENABLE_XROOTD=Off -DCMAKE_CXX_STANDARD=17 -DCMAKE_CXX_EXTENSIONS=Off -DCMAKE_CXX_COMPILER=nvcc_wrapper -DENABLE_GPU=on -DUSE_KOKKOS=ON -DRNDGEN_CPU=Off
  make -j16
  echo "x-x-x-x-x Kokkos Nvidia CURAND BUILD DONE! x-x-x-x-x"
  cd ..
fi
### CPURNG
if [ "$system" = "exalearn5" ]; then
  echo "x-x-x-x-x Kokkos Nvidia CPURNG BUILD x-x-x-x-x"
  mkdir -p build-exalearn5-kokkos-nv-cpurng
  cd build-exalearn5-kokkos-nv-cpurng
  cmake ../FastCaloSimAnalyzer -DENABLE_XROOTD=Off -DCMAKE_CXX_STANDARD=17 -DCMAKE_CXX_EXTENSIONS=Off -DCMAKE_CXX_COMPILER=nvcc_wrapper -DENABLE_GPU=on -DUSE_KOKKOS=ON -DRNDGEN_CPU=On
  make -j16
  echo "x-x-x-x-x Kokkos Nvidia CPURNG BUILD DONE! x-x-x-x-x"
  cd ..
fi


