#!/bin/bash

# set -e  # Exit immediately if a command exits with a non-zero status.
#set -x  # Print commands and their arguments as they are executed.

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
REGISTRY_PROJECT=registry.nersc.gov/m2845
MOUNTED_RUN_DIR=/run
DATAPATH="${FCS_DATAPATH}"

CHECK_LOCAL_IMAGES=${CHECK_LOCAL_IMAGES:-0}

# Ensure the log directory exists
mkdir -p "${LOG_DIR}"

clean() {
    $CONTAINER_CMD ps -aq | xargs -r $CONTAINER_CMD stop | xargs -r $CONTAINER_CMD rm --force
}

check_command_exists() {
    command -v "$1" >/dev/null 2>&1
}

run_fcs_image() {
    local base_image=$1
    local image_type=$2
    local image_tag=$(basename ${base_image})
    image_tag=${image_tag//:/}
    local fcs_image_tag=${REGISTRY_PROJECT}/${image_type}:${ROOT_DOT_VERSION}-${image_tag}
    local container_name="${image_type}_${image_tag}"
    local container_script
    local run_cmd
    local backend
    local container_log_file="run_log_${RUNNER_LABEL}_${image_type}-${ROOT_DOT_VERSION}-${image_tag}_$(date +%Y%m%d%H%M%S).txt"

    case ${image_type} in
    fcs-cuda)
        container_script="${MOUNTED_RUN_DIR}/run_fcs-gpu.sh"
        backend="--gpus 1 "
        ;;
    fcs-kokkos-cuda)
        container_script="${MOUNTED_RUN_DIR}/run_fcs-kokkos-cuda.sh"
        backend="--gpus 1 "
        ;;
    fcs-x86)
        container_script="${MOUNTED_RUN_DIR}/run_fcs-x86.sh"
        backend=""
        fcs_image_tag=${REGISTRY_PROJECT}/fcs:${ROOT_DOT_VERSION}-${image_tag}
        ;;
    fcs-stdpar)
        container_script="${MOUNTED_RUN_DIR}/run_fcs-gpu.sh"
        backend="--gpus 1 "
        fcs_image_tag=${REGISTRY_PROJECT}/fcs-stdpar:${ROOT_DOT_VERSION}-${image_tag}
        ;;
    fcs-hip-cuda)
        container_script="${MOUNTED_RUN_DIR}/run_fcs-gpu.sh"
        backend="--gpus 1 "
        fcs_image_tag=${REGISTRY_PROJECT}/hip-cuda:${ROOT_DOT_VERSION}-${image_tag}
        ;;
    esac

    run_cmd="${CONTAINER_CMD} run \
        --attach STDOUT \
        --rm \
        ${backend}\
        -v $PWD:${MOUNTED_RUN_DIR} \
        -v ${DATAPATH}:/input \
        -v "${LOG_DIR}":/log_dir \
        -e SYSINFO=${MOUNTED_RUN_DIR}/sysinfo.pl \
        -e DATAPATH=${DATAPATH} \
        -e LOGFILE=/log_dir/${container_log_file} \
        ${fcs_image_tag} \
        ${container_script}"

    echo "Current directory: $(pwd)"
    echo "INFO - $(date) - Running image: ${fcs_image_tag}"
    echo "INFO - $(date) - CMD: ${run_cmd}"
    ${run_cmd}
    echo "INFO - $(date) - Finished running image: ${fcs_image_tag}"
}

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

echo "Logging into ${REGISTRY_PROJECT}"
echo "${NERSC_CONTAINER_REGISTRY_PASSWORD}" | $CONTAINER_CMD login -u "${NERSC_CONTAINER_REGISTRY_USER}" --password-stdin ${REGISTRY_PROJECT}

for base_image in ${UBUNTU_BASE_IMAGE} ${NVHPC_BASE_IMAGE} ${CUDA_BASE_IMAGE}; do

    # Run FCS images using different base images
    for image_type in fcs-x86 fcs-cuda fcs-kokkos-cuda; do
        # Only run FCS x86 variant using Ubuntu base image
        if [ "${image_type}" = "fcs-x86" ] && [ "${base_image}" != ${UBUNTU_BASE_IMAGE} ]; then
            continue
        fi
        # Only run FCS GPU variant using non-Ubuntu base image
        if [ "${image_type}" != "fcs-x86" ] && [ "${base_image}" = ${UBUNTU_BASE_IMAGE} ]; then
            continue
        fi
        run_fcs_image ${base_image} ${image_type}
    done
done

run_fcs_image ${NVHPC_BASE_IMAGE} fcs-stdpar

# run_fcs_image ${CUDA_BASE_IMAGE} fcs-hip-cuda

clean
