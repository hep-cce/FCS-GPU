FROM dingpf/fcs-rocm

USER root

RUN \
  cd /hep-mini-apps && \
  mkdir -p llvm-amdgpu && \
  git clone --depth 1 --branch llvmorg-19.1.0 https://github.com/llvm/llvm-project.git && \
  cd llvm-project && \
  mkdir -p build && \
  cd build && \
  cmake -G "Unix Makefiles" \
    -B /hep-mini-apps/llvm-project/build/  \
    -DLLVM_ENABLE_PROJECTS="clang;clang-tools-extra;lld;lldb;compiler-rt" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILER=g++ \
    -DLLVM_ENABLE_RUNTIMES:STRING="openmp;offload" \
    -DCLANG_DEFAULT_OPENMP_RUNTIME:STRING=libomp \
    -DCMAKE_INSTALL_PREFIX=/hep-mini-apps/llvm-amdgpu \
    -DLLVM_TARGETS_TO_BUILD:STRING="X86;AMDGPU" \
    -DLIBOMPTARGET_DEVICE_ARCHITECTURES="gfx906;gfx908;gfx90a" \
    /hep-mini-apps/llvm-project/llvm && \
  make -j128 && \
  make install && \
  cd ../../ && \
  rm -rf llvm-project
