This directory contains scripts to run on cori.nersc.gov GPU nodes

Setup
-----

module load gcc/8.3.0
module load cuda
export ROOTDIR=/global/cfs/cdirs/atlas/leggett/root/v6-14-08_gcc83
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$ROOTDIR/lib

## FastCaloSim built in $BLDDIR, configured with input data in
## /dev/shm/${USER}/FastCaloSimInputs
source $BLDDIR/x86_64-linux15-gcc8-opt/setup.sh

## to get a 30 min interactive whole node session on the GPU nodes:
module load esslurm
salloc -N 1 -t 30 -c 80 --gres=gpu:8 --exclusive -C gpu -A m1759


Running
-------

FastCaloSim command that will be run is "runTFCSSimulation"

run.sh: script that will start a number of concurrent FastCaloSim
jobs. 

mrun.sh: script that will call run.sh a number of times to run a
sequence of jobs. It will startup an nvidia-cuda-mps-control server.

Input data is assumed to be in /dev/shm/${USER}/FastCaloSimInputs


To run just once with N concurrent jobs, and no nvidia-cuda-mps-control:
> srun -n1 -N1 ./run.sh N
This requires the input data to already be in /dev/shm/${USER}/FastCaloSimInputs

To run using nvidia-cuda-mps-control with N concurrent jobs:
> srun -n1 -N1 ./mrun.sh N
This will copy the input data to /dev/shm/${USER}/FastCaloSimInputs if
it's not alread there

To run a sequence of jobs, using nvidia-cuda-mps-control, with M to N
concurrent jobs:
> srun -n1 -N1 ./mrun.sh M N

To run a sequence of jobs, using nvidia-cuda-mps-control, with 1 to
NPROC concurrent jobs:
> srun -n1 -N1 ./mrun.sh


Output
------
Each concurrent job writes its output to directories "r_N".

STDOUT is a CSV of:
n_concurrent_jobs
Time of Chain_0
Time of EventLoop
Time of EventLoop GPU Chain A
Time of EventLoop GPU Chain B
Time of EventLoop Host Chain A
Time of EventLoop before chain simul
total job time

these are extracted from the log files in r_N/run.log by the script
parselogs.pl. They are AVERAGED over all concurrent jobs in a run.


Customization
--------------

- Turn off nvidia-cuda-mps-control:
in mrun.sh, change USE_MPS_SERVER to 0

- To change the GPU used (default is GPU 0):
in mrun.sh, change CUDA_VISIBLE_DEVICES to desired GPU

- Record output of nvidia-smi every N seconds (default is don't record):
in mrun.sh, change SMI_INTERVAL to desired number of seconds

- Change sequence of CPU cores:
in run.sh, change CORELAYOUT
   0 : core 0 non HT, core 1 non HT, core 0 HT, core 1 HT
   1 : alternate cores, non HT, then HT: 0,1,2,3....
   2 : core 0 HT, non-HT, core 1 HT, non-HT
mapping between coreid and concurrent job is recorded in "coreid.log"
full run command is in "job.log"

- to do a dry run without doing anything
in run.sh, change TEST to 1

Notes
-----

These scripts should be usable on machines other than cori, but would
require a bit of tweaking if they are not dual cpu ( for COREID
calculations ), and if srun is not being used (starting and stopping
nvidia-cuda-mps-control requires sudo)
