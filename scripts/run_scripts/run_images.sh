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
DATAPATH=/global/cfs/cdirs/atlas/leggett/data/FastCaloSimInputs:/input
MOUNTED_RUN_DIR=/run
LOG_DIR=run_logs

CHECK_LOCAL_IMAGES=${CHECK_LOCAL_IMAGES:-0}

mkdir -p ${LOG_DIR}

logfile="${LOG_DIR}/run_image_log_$(date +%Y%m%d%H%M%S).txt"

clean() {
    podman-hpc ps -aq | xargs -r podman-hpc stop | xargs -r podman-hpc rm --force >>${logfile} 2>&1
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
    local container_log_file="${LOG_DIR}/run_log_${image_type}-${ROOT_DOT_VERSION}-${image_tag}_$(date +%Y%m%d%H%M%S).txt"

    case ${image_type} in
    fcs-cuda)
        container_script="${MOUNTED_RUN_DIR}/run_fcs-gpu.sh"
        backend="--gpu "
        ;;
    fcs-kokkos-cuda)
        container_script="${MOUNTED_RUN_DIR}/run_fcs-kokkos-cuda.sh"
        backend="--gpu "
        ;;
    fcs-cpu)
        container_script="${MOUNTED_RUN_DIR}/run_fcs-cpu.sh"
        backend=""
        ;;
    fcs-stdpar)
        container_script="${MOUNTED_RUN_DIR}/run_fcs-gpu.sh"
        backend="--gpu "
        ;;
    fcs-hip-cuda)
        container_script="${MOUNTED_RUN_DIR}/run_fcs-gpu.sh"
        backend="--gpu "
        ;;
    esac

    run_cmd="${CONTAINER_CMD} run \
        --rm \
        ${backend}\
        -v $PWD:${MOUNTED_RUN_DIR} \
        -v ${DATAPATH} \
        -e LOGFILE=${MOUNTED_RUN_DIR}/${container_log_file} \
        -e DATAPATH=${DATAPATH} \
        -e SYSINFO=${MOUNTED_RUN_DIR}/sysinfo.pl
        ${fcs_image_tag} \
        ${container_script}"

    echo "Current directory: $(pwd)" >>${logfile} 2>&1
    echo "INFO - $(date)) - Running image: ${fcs_image_tag}" >>${logfile} 2>&1
    echo "INFO - $(date)) - CMD: ${run_cmd}" >>${logfile} 2>&1
    eval ${run_cmd} >>${logfile} 2>&1
}

if check_command_exists podman-hpc; then
    CONTAINER_CMD="podman-hpc"
elif check_command_exists docker; then
    CONTAINER_CMD="docker"
else
    echo "ERROR: Neither podman-hpc nor docker is installed on this system." >>${logfile} 2>&1
    exit 1
fi

echo "Logging into ${NERSC_CONTAINER_REGISTRY_URL}"
echo "${NERSC_CONTAINER_REGISTRY_PASSWORD}" | \
    $CONTAINER_CMD login -u "${NERSC_CONTAINER_REGISTRY_USER}" \
    --password-stdin ${CONTAINER_REGISTRY_URL}

for base_image in ${UBUNTU_BASE_IMAGE} ${NVHPC_BASE_IMAGE} ${CUDA_BASE_IMAGE}; do

    # Run FCS images using different base images
    for image_type in fcs-cpu fcs-cuda fcs-kokkos-cuda; do
        # Only run FCS CPU variant using Ubuntu base image
        if [ "${image_type}" = "fcs-cpu" ] && [ "${base_image}" != ${UBUNTU_BASE_IMAGE} ]; then
            continue
        fi
        # Only run FCS GPU variant using non-Ubuntu base image
        if [ "${image_type}" != "fcs-cpu" ] && [ "${base_image}" = ${UBUNTU_BASE_IMAGE} ]; then
            continue
        fi
        run_fcs_image ${base_image} ${image_type}
    done
done

run_fcs_image ${NVHPC_BASE_IMAGE} fcs-stdpar

run_fcs_image ${CUDA_BASE_IMAGE} fcs-hip-cuda

clean
