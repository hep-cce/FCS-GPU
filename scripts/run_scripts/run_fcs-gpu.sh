#!/bin/bash

logfile=${LOGFILE:-/tmp/default_logfile.txt}

log() {
    local message="$1"
    echo "INFO - $(date) - ${message}" >> ${logfile} 2>&1
}

log_command() {
    local command="$1"
    echo "INFO - $(date) - Executing: ${command}" >> ${logfile} 2>&1
    eval ${command} >> ${logfile} 2>&1
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
log_command "export FCS_DATAPATH=/input"
log_command "source /hep-mini-apps/root/install/bin/thisroot.sh"
log_command "source /hep-mini-apps/FCS-GPU/install/setup.sh"

log "TFCSSimulation"
log_command "runTFCSSimulation --earlyReturn --energy 65536"
