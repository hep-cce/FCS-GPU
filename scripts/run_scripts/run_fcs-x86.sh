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
    eval ${command} 2>&1 | tee -a ${logfile} || {
        echo "ERROR - $(date) - Command failed: ${command}" | tee -a ${logfile}
        exit 1
    }
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

check_command_exists() {
    command -v "$1" >/dev/null 2>&1
}

log "DATAPATH: $DATAPATH"

log_info "perl ${SYSINFO}" "System"
log_info "lscpu" "CPU"

log "Setup"
export FCS_DATAPATH=/input
source /hep-mini-apps/root/install/bin/thisroot.sh
source /hep-mini-apps/FCS-GPU/install/setup.sh

log "TFCSSimulation"
log_command "runTFCSSimulation --earlyReturn --energy 65536"
