BUild the code:

module load cmake gcc/6.4.0 cuda/9.0 root/v6.14.08-gcc-6.4.0

mkdir build 
cd build
cmake ../FastCaloSimAnalyzer -DENABLE_XROOTD=off -DENABLE_GPU=on -DINPUT_PATH="/hpcgpfs01/work/csi/cce/FastCaloSimInputs"
make -j 8

. x86_64-slc7-gcc64-opt/setup.sh

#to your work directory

cd $your_work_directy
srun -A cce -p long -N1 -n1 runTFCSShapeValidation
