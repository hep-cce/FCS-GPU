#!/bin/bash

#SBATCH --partition=long
#SBATCH --qos=normal
#SBATCH --account=cce
###SBATCH --gres=gpu:1
###SBATCH --constraint=pascal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=00:30:00
#SBATCH --job-name=testfcs
###SBATCH --mail-user=fmohammad@bnl.gov
###SBATCH --mail-type=ALL

##module load cmake gcc/8.2.0 cuda/10.1 root/v6.18.02-gcc-8.2.0
module purge
module load gcc/8.2.0 llvm/13.0.0 root/v6.20.04-gcc-8.2.0

. x86_64-slc7-clang130-opt/setup.sh

export OMP_NUM_THREADS=1

srun -n 1 ./x86_64-slc7-clang130-opt/bin/runTFCSSimulation > out.log

module purge
exit
