#!/bin/bash

#SBATCH --partition=long
#SBATCH --qos=normal
#SBATCH --account=cce
#SBATCH --gres=gpu:2
#SBATCH --constraint=pascal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=00:30:00
#SBATCH --job-name=testfcs
###SBATCH --mail-user=fmohammad@bnl.gov
###SBATCH --mail-type=ALL

#export OMP_NUM_THREADS=1

srun -n 1 ./x86_64-slc7-clang130-opt/bin/runTFCSSimulation > out.log
##srun -n 1 ./x86_64-slc7-gcc8-opt/bin/runTFCSSimulation > out.log

exit
