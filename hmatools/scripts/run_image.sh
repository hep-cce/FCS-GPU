#!/bin/bash

set -e  # Exit immediately if a command exits with a non-zero status.
set -x  # Print commands and their arguments as they are executed.

# Ensure the directories exists
mkdir -p "${LOG_DIR}"
mkdir -p "${OUTPUT_DIR}"

logfile="postprocess_log_$(date +%Y%m%d%H%M%S).txt"

script="${SCRIPT}"
input_dir="${INPUT_DIR}"
output_dir="${OUTPUT_DIR}"

check_command_exists() {
  echo "Checking if command $1 exists..."
  command -v "$1" >>${logfile} 2>&1
}

echo "Starting script..."

if check_command_exists podman-hpc; then
  CONTAINER_CMD="podman-hpc"
elif check_command_exists docker; then
  CONTAINER_CMD="docker"
else
  echo "ERROR: Neither podman-hpc nor docker is installed on this system."
  exit 1
fi

echo "Using container command: ${CONTAINER_CMD}"

echo "INFO - $(date) - Run postprocessing"
$CONTAINER_CMD run \
    --attach STDOUT \
    --rm \
    -v $PWD:/workspace \
    -v $input_dir:/input \
    -v $output_dir:/output \
    -v "${LOG_DIR}":/log_dir \
    -w /workspace \
    -e LOGFILE=/log_dir/${logfile} \
    hmatools \
    ${script} \
| tee -a ${logfile}

echo "Script completed successfully!"