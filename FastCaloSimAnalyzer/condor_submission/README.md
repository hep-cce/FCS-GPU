# Submit jobs in lxplus condor 

`Job submission is only available from LXPLUS nodes`



``` 
    $ cd /path/to/FastCaloSimAnalyzer/condor_submission/
    $ python energy_shape_parametrization_condor_submit.py -i sample_list/dsid_list_central_pdgid22.txt

```


 _Following options can be set to the job submission script :_

```
# user information
email = "a.hasib@cern.ch"

# set general parameters
memory = 14000
flavor = "espresso"

metadata = "inputSampleList.txt"
outDir = "/eos/atlas/user/a/ahasib/public/Simul-FastCalo/ParametrizationProductionVer01/"
version = "ver01"

runEpara_temp = "runEpara_temp.C"
runShape_temp = "runShape_temp.C"
sub_temp = "template_submit.sub"
exe_temp = "template_run.sh"


# energy parametrization related parameters
npca_primary = 5
npca_secondary = 1
do_validation = 0

init_epara = "init_epara.C"
run_epara = "run_epara.C"

# shape parametrization related parameters
energy_cutoff = 0.9995
do_ACside = 1
do_Aside = 0
do_Cside = 0

init_shape = "initTFCS2DParametrizationHistogram.C"
run_shape  = "runTFCS2DParametrizationHistogram.C"
```