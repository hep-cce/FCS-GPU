#!/bin/bash

logfile=${LOGFILE}

log_command() {
    local command="$1"
    echo "INFO - $(date) - Executing: ${command}" | tee -a ${logfile}
    eval ${command} | tee -a ${logfile}
}

log_command "plot -i /input -o /output"
