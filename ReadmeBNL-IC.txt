BUild the code:

do
module load cmake gcc/6.4.0 cuda/9.0 root/v6.14.08-gcc-6.4.0
or 
module load cmake gcc/8.2.0 cuda/10.1 root/6.18.02-gcc-8.2.0

mkdir build 
cd build
cmake ../FastCaloSimAnalyzer -DENABLE_XROOTD=off -DENABLE_GPU=on
make -j 8

. x86_64-slc7-gcc8-opt/setup.sh

#to your work directory

cd $your_work_directy
srun -A cce -p long -N1 -n1 runTFCSSimulation --dataDir=/hpcgpfs01/work/csi/cce/FastCaloSimInputs
