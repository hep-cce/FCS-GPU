#!/bin/bash

batchDir=${PWD}/condor_submission/batch/
memory=2000
email=petr.jacka@cern.ch
jobname=zvertexStudies

dsid_zv0=100035
dsids=(100032 100033 100034 100035 100036 100037 100038)
##dsids=(100038)


#dsid_zv0=100042
#dsids=(100039 100040 100041 100042 100043 100044 100045)

#dsid_zv0=100049
#dsids=(100046 100047 100048 100049 100050 100051 100052)

#dsid_zv0=100056
#dsids=(100053 100054 100055 100056 100057 100058 100059)

#dsid_zv0=100063
#dsids=(100060 100061 100062 100063 100064 100065 100066)


#dsid_zv0=431696
#dsids=(431696)

#dsid_zv0=431697
#dsids=(431697)

#dsid_zv0=431698
#dsids=(431698)

#dsid_zv0=431699
#dsids=(431699)


#### Run this to produce <weight> histograms for extrapolator

#do2Dparameterization=1
#isPhisymmetry=1
#doMeanRz=1
#useMeanRz=0
#doZVertexStudies=0

#./condor_submission/simpleJob.py $batchDir $memory $email $jobname ./condor_submission/runZvertexStudies.sh $dsid_zv0 $dsid_zv0 $do2Dparameterization $isPhisymmetry $doMeanRz $useMeanRz $doZVertexStudies

#### -----------------------------------------------------------


#### Loop over shifted samples


do2Dparameterization=1
isPhisymmetry=1
doMeanRz=0
useMeanRz=1
doZVertexStudies=1

for dsid in ${dsids[@]}
do
   ./condor_submission/simpleJob.py $batchDir $memory $email $jobname ./condor_submission/runZvertexStudies.sh $dsid $dsid_zv0 $do2Dparameterization $isPhisymmetry $doMeanRz $useMeanRz $doZVertexStudies
done



###################################################

#for dsid in `seq 431000 431009`;
#do
   ##echo $dsid
   #./condor_submission/simpleJob.py $batchDir $memory $email $jobname ./condor_submission/runZvertexStudies.sh $dsid $dsid $do2Dparameterization $isPhisymmetry $doMeanRz $useMeanRz $doZVertexStudies
#done




