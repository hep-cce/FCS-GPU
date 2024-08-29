#!/bin/bash

set -e  # Exit immediately if a command exits with a non-zero status.
set -x  # Print commands and their arguments as they are executed.

check_command_exists() {
  echo "Checking if command $1 exists..."
  command -v "$1"
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

echo "INFO - $(date) - Building hmatools"
$CONTAINER_CMD build -f ../hmatools.Dockerfile \
  -t hmatools \
  .. # Context path: self_hosted_runner/hmatools
echo "INFO - $(date) - Built hmatools"

echo "Script completed successfully!"
