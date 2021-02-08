if [ -z $ATLAS_LOCAL_ASETUP_VERSION ]; 
then
   # do it only if it has not been done yet
   export ATLAS_LOCAL_ROOT_BASE=/cvmfs/atlas.cern.ch/repo/ATLASLocalRootBase
   alias setupATLAS='source ${ATLAS_LOCAL_ROOT_BASE}/user/atlasLocalSetup.sh'
   source $ATLAS_LOCAL_ROOT_BASE/user/atlasLocalSetup.sh
else
   echo "Atlas environment already configured."
fi
lsetup "root 6.14.04-x86_64-slc6-gcc62-opt"
lsetup git 
