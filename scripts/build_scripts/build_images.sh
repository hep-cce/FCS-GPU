#!/bin/bash

OS_BASE=ubuntu22.04
CUDA_VERSION=12.2.2-devel-${OS_BASE}
NVHPC_VERSION=23.9-devel-cuda12.2-${OS_BASE}
ROOT_VERSION=v6-30-04
FCS_BRANCH=dingpf/packaging

ROOT_DOT_VERSION=${ROOT_VERSION//v/}
ROOT_DOT_VERSION=${ROOT_DOT_VERSION//-/.}
NVHPC_BASE_IMAGE=nvcr.io/nvidia/nvhpc:${NVHPC_VERSION}
CUDA_BASE_IMAGE=docker.io/nvidia/cuda:${CUDA_VERSION}
UBUNTU_BASE_IMAGE=docker.io/library/ubuntu:22.04
#REGISTRY_PROJECT=docker.io/dingpf
REGISTRY_PROJECT=registry.nersc.gov/m2845

CHECK_REPO=${CHECK_REPO:-0}

logfile="build_image_log_$(date +%Y%m%d%H%M%S).txt"

check_image_exists() {
  local image_tag=$1

  podman-hpc manifest inspect ${image_tag} > /dev/null 2>&1
  if [ $? -eq 0 ]; then
    return 0
  else
    return 1
  fi
}

build_and_push_root_image() {
  local base_image=$1
  local image_tag=$(basename ${base_image})
  image_tag=${image_tag//:/}
  local root_image_tag=${REGISTRY_PROJECT}/root:${ROOT_DOT_VERSION}-${image_tag}

  if [ ${CHECK_REPO} -eq 1 ]; then
    check_image_exists ${root_image_tag}
    if [ $? -eq 0 ]; then
      echo "INFO - $(date) - Skipping build. Image ${root_image_tag} already exists in the registry."
      return
    fi
  fi

  echo "INFO - $(date)) - Building image: ${root_image_tag}"
  podman-hpc build -f root.Dockerfile \
    --build-arg=BASE=${base_image} \
    --build-arg=ROOT_VERSION=${ROOT_VERSION} \
    -t ${root_image_tag} . >> ${logfile}
  echo "INFO - $(date) - Pushing image: ${root_image_tag}"
  podman-hpc push ${root_image_tag}
}

build_and_push_fcs_image() {
  local base_image=$1
  local image_type=$2
  local image_tag=$(basename ${base_image})
  image_tag=${image_tag//:/}
  local root_image_tag=${REGISTRY_PROJECT}/root:${ROOT_DOT_VERSION}-${image_tag}
  local fcs_image_tag=${REGISTRY_PROJECT}/${image_type}:${ROOT_DOT_VERSION}-${image_tag}

  if [ ${CHECK_REPO} -eq 1 ]; then
    check_image_exists ${fcs_image_tag}
    if [ $? -eq 0 ]; then
      echo "INFO - $(date) - Image ${fcs_image_tag} already exists in the registry. Skipping build."
      return
    fi
  fi

  echo "INFO - $(date)) - Building image: ${fcs_image_tag}"
  podman-hpc build -f ${image_type}.Dockerfile \
    --build-arg=BASE=${root_image_tag} \
    --build-arg=FCS_BRANCH=${FCS_BRANCH} \
    -t ${fcs_image_tag} . >> ${logfile}
  echo "INFO - $(date) - Pushing image: ${fcs_image_tag}"
  podman-hpc push ${fcs_image_tag}
}

for base_image in ${UBUNTU_BASE_IMAGE} ${NVHPC_BASE_IMAGE} ${CUDA_BASE_IMAGE}; do
  build_and_push_root_image ${base_image}

  # Build FCS images using different base images
  for image_type in fcs fcs-cuda fcs-kokkos-cuda; do
     # Only build FCS CPU variant using Ubuntu base image
     if [ "${image_type}" = "fcs" ] && [ "${base_image}" != ${UBUNTU_BASE_IMAGE} ]; then
      continue
     fi
     # Only build FCS GPU variant using non-Ubuntu base image
     if [ "${image_type}" != "fcs" ] && [ "${base_image}" = ${UBUNTU_BASE_IMAGE} ]; then
      continue
     fi
      build_and_push_fcs_image ${base_image} ${image_type}
  done
done
