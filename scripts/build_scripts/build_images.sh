#!/bin/bash

set -e  # Exit immediately if a command exits with a non-zero status.
set -x  # Print commands and their arguments as they are executed.

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

logfile="${LOG_DIR}/build_image_log_$(date +%Y%m%d%H%M%S).txt"

# Ensure the log directory exists
mkdir -p "${LOG_DIR}"

check_command_exists() {
  echo "Checking if command $1 exists..." | tee -a ${logfile}
  command -v "$1" >>${logfile} 2>&1
}

check_image_exists() {
  local image_tag=$1

  echo "Checking if image ${image_tag} exists..." | tee -a ${logfile}
  $CONTAINER_CMD manifest inspect ${image_tag} >>${logfile} 2>&1
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
      echo "INFO - $(date) - Skipping build. Image ${root_image_tag} already exists in the registry." | tee -a ${logfile}
      return
    fi
  fi

  echo "INFO - $(date) - Building image: ${root_image_tag}" | tee -a ${logfile}
  $CONTAINER_CMD build -f root.Dockerfile \
    --build-arg=BASE=${base_image} \
    --build-arg=ROOT_VERSION=${ROOT_VERSION} \
    -t ${root_image_tag} . | tee -a ${logfile}

  # echo "INFO - $(date) - Pushing image: ${root_image_tag}" | tee -a ${logfile}
  # $CONTAINER_CMD push ${root_image_tag} | tee -a ${logfile}
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
      echo "INFO - $(date) - Image ${fcs_image_tag} already exists in the registry. Skipping build." | tee -a ${logfile}
      return
    fi
  fi

  echo "INFO - $(date) - Building image: ${fcs_image_tag}" | tee -a ${logfile}
  $CONTAINER_CMD build -f ${image_type}.Dockerfile \
    --build-arg=BASE=${root_image_tag} \
    --build-arg=FCS_BRANCH=${FCS_BRANCH} \
    -t ${fcs_image_tag} \
    . | tee -a ${logfile}

  # echo "INFO - $(date) - Pushing image: ${fcs_image_tag}" | tee -a ${logfile}
  # $CONTAINER_CMD push ${fcs_image_tag} | tee -a ${logfile}
}

echo "Starting script..." | tee -a ${logfile}

if check_command_exists podman-hpc; then
    echo "Using podman-hpc"
    CONTAINER_CMD="podman-hpc"
elif check_command_exists docker; then
    echo "Using docker"
    CONTAINER_CMD="docker"
else
    echo "ERROR: Neither podman-hpc nor docker is installed on this system."
    exit 1
fi

echo "Using container command: ${CONTAINER_CMD}" | tee -a ${logfile}

for base_image in ${UBUNTU_BASE_IMAGE} ${NVHPC_BASE_IMAGE} ${CUDA_BASE_IMAGE}; do
  echo "Processing base image: ${base_image}" | tee -a ${logfile}
  build_and_push_root_image ${base_image}

  # Build FCS images using different base images
  for image_type in fcs-x86 fcs-cuda fcs-kokkos-cuda; do
    # Only build FCS x86 variant using Ubuntu base image
    if [ "${image_type}" = "fcs-x86" ] && [ "${base_image}" != ${UBUNTU_BASE_IMAGE} ]; then
      echo "Skipping FCS x86 variant for non-Ubuntu base image" | tee -a ${logfile}
      continue
    fi
    # Only build FCS GPU variant using non-Ubuntu base image
    if [ "${image_type}" != "fcs-x86" ] && [ "${base_image}" = ${UBUNTU_BASE_IMAGE} ]; then
      echo "Skipping FCS GPU variant for Ubuntu base image" | tee -a ${logfile}
      continue
    fi
    echo "Building FCS image: ${image_type} with base image: ${base_image}" | tee -a ${logfile}
    build_and_push_fcs_image ${base_image} ${image_type}
  done
done

echo "Building additional FCS images..." | tee -a ${logfile}
build_and_push_fcs_image ${NVHPC_BASE_IMAGE} fcs-stdpar-cuda
build_and_push_fcs_image ${CUDA_BASE_IMAGE} fcs-hip-cuda

echo "Script completed successfully!" | tee -a ${logfile}
