#!/bin/bash

OS_BASE=ubuntu22.04
CUDA_VERSION=12.2.2-devel-${OS_BASE}
NVHPC_VERSION=23.9-devel-cuda12.2-${OS_BASE}
ROOT_VERSION=v6-30-04
ROOT_DOT_VERSION=${ROOT_VERSION//v/}
ROOT_DOT_VERSION=${ROOT_DOT_VERSION//-/.}

NVHPC_BASE_IMAGE=nvcr.io/nvidia/nvhpc:${NVHPC_VERSION}
CUDA_BASE_IMAGE=docker.io/nvidia/cuda:${CUDA_VERSION}
REGISTRY_PROJECT=docker.io/dingpf

for base_image in ${NVHPC_BASE_IMAGE} ${CUDA_BASE_IMAGE}; do
  image_tag=$(basename ${base_image})
  image_tag=${image_tag//:/} # example: nvhpc23.9-devel-cuda12.2-ubuntu22.04

  root_image_tag=${REGISTRY_PROJECT}/root:${ROOT_DOT_VERSION}-${image_tag} # example: dingpf/root:6.30.04-nvhpc23.9-devel-cuda12.2-ubuntu22.04
  podman-hpc build -f root.Dockerfile \
    --build-arg=BASE=${base_image} \
    --build-arg=ROOT_VERSION=${ROOT_VERSION} \
    -t ${root_image_tag} .

  for image_type in fcs-cuda fcs-kokkos-cuda; do
    fcs_image_tag=${REGISTRY_PROJECT}/${image_type}:${ROOT_DOT_VERSION}-${image_tag} # example: dingpf/fcs-cuda:6.30.04-nvhpc23.9-devel-cuda12.2-ubuntu22.04
    podman-hpc build -f ${image_type}.Dockerfile \
      --build-arg=BASE=${root_image_tag} \
      -t ${fcs_image_tag} .
  done
done