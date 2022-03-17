#!/bin/bash

TEST=0

## The FastCaloSim program to run
CMD=runTFCSSimulation

## CORELAYOUT:
## 0 : core 0 non HT, core 1 non HT, core 0 HT, core 1 HT
## 1 : alternate cores, non HT, then HT: 0,1,2,3....
## 2 : core 0 HT, non-HT, core 1 HT, non-HT
CORELAYOUT=1

if [ "$#" -eq 0 ]; then
    NPROC=1
else
    NPROC=$1
fi

if [[ $TEST -eq 0 ]]; then
    echo -n "$NPROC"
fi

rm -rf r_*

for n in `seq 1 $NPROC`; do
    rm -rf r_$n
    mkdir -p r_$n
done

NCPS=$( lscpu | grep '^Core' | awk '{print $NF}' )
NUMCPU=$( lscpu | grep '^Socket' | awk '{print $NF}' )
NTPC=$( lscpu | grep '^Thread' | awk '{print $NF}' )
NUMCORESperCPU=$(( $NCPS * $NTPC ))

# echo $NCPS $NUMCORESperCPU

CWD=`pwd -P`
TIMELOG="$CWD/time.log"
rm -f $TIMELOG

rm -f $CWD/job.log
for n in `seq 1 $NPROC`; do
    DIR=${CWD}/r_${n}
    cd $DIR
    if [[ $CORELAYOUT -eq 0 ]]; then
        if [[ $n -le $NCPS ]]; then
            CORE=$(( ($n-1) * $NTPC ))
        elif [[ $n -le $(( $NCPS*2 )) ]]; then
            CORE=$(( ($n-$NCPS-1) * $NTPC + 1 ))
        elif [[ $n -le $(( $NCPS*3 )) ]]; then
            CORE=$(( $NCPS*$NUMCPU + ($n-2*$NCPS-1)*$NTPC ))
        else
            CORE=$(( $NCPS*$NUMCPU + ($n-3*$NCPS-1)*$NTPC + 1 ))
        fi
    elif [[ $CORELAYOUT -eq 1 ]]; then
        CORE=$(( $n - 1 ))
    elif [[ $CORELAYOUT -eq 2 ]]; then
        if [[ $n -gt $NUMCORESperCPU ]]; then
            CORE=$(($n * $NTPC - ($NUMCORESperCPU * 2) - 1))
        else
            CORE=$(($n * $NTPC - 2))
        fi
    else
        echo "   ERROR: unexepected value of CORELAYOUT = $CORELAYOUT"
        exit 1
    fi
    TS="taskset -c $CORE"
    COREID[$n]=$CORE
    FCMD="/usr/bin/time --append -o $TIMELOG -f %e $TS $CMD"
    if [[ $TEST -eq 0 ]]; then
        echo "$FCMD" >> $CWD/job.log
        $FCMD >& run.log &
        PID=$!
    else
        echo " $n -> $FCMD"
        PID=$(( 999000 + $n ))
    fi
    JOBID[$PID]=$n
done

cd $CWD

if [[ $TEST -eq 0 ]]; then
    sleep 10
fi
./getprocspeed.pl > $CWD/corespeed.log

rm -f $CWD/coreid.log
for i in "${!COREID[@]}"; do
    echo "$i ${COREID[$i]}" >> $CWD/coreid.log
done

if [[ $TEST -ne 0 ]]; then
    exit 0
fi


FAIL=0
FAILJOB=()
for job in `jobs -p`; do
#    echo "wait $job"
    wait $job || FAILJOB=( $FAILJOB $job )
done

if [[ ${#FAILJOB[@]} -ne 0 ]]; then
    echo "  --> ${#FAILJOB[@]} jobs failed: "
    FAIL=1
    for (( i=0; i<${#FAILJOB[@]}; i++ )); do
        echo "    FAILURE in job PID: ${FAILJOB[$i]}   id: ${JOBID[${FAILJOB[$i]}]}"
        R=${JOBID[${FAILJOB[$i]}]}
        tail -3 r_$R/run.log
        echo " --------------- "
done
fi

if [[ $FAIL -eq 1 ]]; then
    exit 1
fi

sleep 1

./parselogs.pl


exit 0
