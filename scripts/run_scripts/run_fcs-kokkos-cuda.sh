#!/bin/bash

logfile=${LOGFILE}

echo "Logfile: ${logfile}" | tee -a ${logfile}  # Debug line to check LOGFILE value

log() {
    local message="$1"
    echo "INFO - $(date) - ${message}" | tee -a ${logfile}
}

log_command() {
    local command="$1"
    echo "INFO - $(date) - Executing: ${command}" | tee -a ${logfile}
    eval ${command} 2>&1 | tee -a ${logfile}
}

check_command_exists() {
    command -v "$1" >/dev/null 2>&1
}

log_info() {
    local command=$1
    local description=$2
    local fallback_message="No ${description} information found. ${command} is not available."

    if check_command_exists $command; then
        log "${description} information:"
        log_command "$command"
    else
        log "$fallback_message"
    fi
}

log "DATAPATH: $DATAPATH"

log_info "perl ${SYSINFO}" "System"
log_info "lscpu" "CPU"
log_info "nvidia-smi --query-gpu=name,driver_version,count,clocks.max.sm,clocks.max.memory,memory.total,memory.used,memory.free,utilization.gpu,utilization.memory,temperature.gpu,temperature.memory --format=csv" "GPU"

log "Setup"
export FCS_DATAPATH=/input
export LD_LIBRARY_PATH=/hep-mini-apps/Kokkos/install/lib:$LD_LIBRARY_PATH
source /hep-mini-apps/root/install/bin/thisroot.sh
source /hep-mini-apps/FCS-GPU/install/setup.sh

log "TFCSSimulation"
log_command "runTFCSSimulation --earlyReturn --energy 65536"
