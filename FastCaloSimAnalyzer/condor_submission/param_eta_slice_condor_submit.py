#!/usr/bin/evn python
from __future__ import print_function
import sys
import os
import glob
import fileinput
import socket
import commands
import ROOT
import numpy as np
from argparse import ArgumentParser
from pprint import pprint
#--------------------------------------------------
# user information
email = "a.hasib@cern.ch"

# set general parameters
memory = 2000
# flavor = "tomorrow"
MaxRuntime = 1*60*60

outDir = "/eos/atlas/user/a/ahasib/public/Simul-FastCalo/ParametrizationProductionVer06/TFCSParamEtaSlice_HEC_4xup/"

init_file = "initTFCSAnalyzer.C"
run_ParamEtaSlice ="runTFCSCreateParamEtaSlice.cxx"
run_CreateParam = "runTFCSCreateParametrization.cxx"
runParamEtaSlice_temp = "runParamEtaSlice_temp.C"
sub_temp = "template_submit.sub"
exe_temp = "template_ParamEtaSlice_run.sh"
database = "db.txt"
top_dir = "runParamEtaSlice_HEC_up/"
#------------------------------------------------------

Emin = 64
Emax = 4194304

eta_start = 0
eta_stop = 5.

pdgids = [22, 211, 11]
# pdgids = [211, 11]
# pdgids = [22]
# pdgids = [11]
# pdgids = [211]


# electron eta => 3.7 Emin = 256
# electron eta=> 4.9 Emin = 512
# pion eta => 4.25 Emin = 256
# photon eta => 3.4 Emin = 256

#----------------------------------------------------------

def getArguments():
    """Get arguments from command line."""
    parser = ArgumentParser()
    parser.add_argument("-i", "--input", nargs='+', help="Text files containing a list of DSIDs you want to submit")
    parser.add_argument("-c", "--checkOnly", action='store_true', help="Flag for only checking if the condor jobs ran successfuly and produced the parametrizations. You would need to provide a list of DSIDs.")
    parser.add_argument("-w", "--writeFile", default=None, help="Write a txt file with DSIDs of jobs that are broken.")
    return parser.parse_args()

def getListOfDSIDs(inFile):
    dsids=[]
    if inFile:
        with open(inFile) as DSIDList:
            for DSID in DSIDList:
                DSID = DSID.strip("\n")
                dsids.append(DSID)

    return dsids

def checkParamEtaSlices(destination):
    failed_jobs = {}
    etas = np.arange(eta_start, eta_stop, .05)
    for pdgid in pdgids:
        for eta in etas:
            int_etamin = int(eta*100)
            int_etamax = int_etamin + 5
            str_eta = "eta_"+str(int_etamin)+"_"+str(int_etamax)
            dataset = "SelEkin_id"+str(pdgid)+"_Mom"+str(Emin)+"_"+str(Emax)+"_"+str_eta+".root"
            fileNames = commands.getoutput("ls {destination}/{dataset}".format(destination=destination,dataset=dataset))
            for fileName in fileNames.split("\n"):
            # print(fileName)
                if "cannot" in fileNames or not fileNames:
                    failed_jobs[dataset] = "file doesn't exist"
                else:
                   fin = ROOT.TFile(fileName)
                   if fin.IsZombie():
                        failed_jobs[dataset] = "file corrupted"
                        fin.Close()
    return failed_jobs



def main():

    options = getArguments()

    # check to see if running on LXPLUS
    try:
        host = socket.gethostname()
        if not 'lxplus' in host :
            raise Exception
    except Exception:
        print("Condor jobs have to be submitted from a LXPLUS node")
        sys.exit(1)
        # for each input dsid lists
    if not options.checkOnly:

        print("Running Condor Submission ...")
        etas = np.arange(eta_start, eta_stop, .05)
        # print(etas)
        for pdgid in pdgids:
            # for each dsid copy, modify all files and submit
            for eta in etas:
                int_etamin = int(eta*100)
                int_etamax = int_etamin + 5
                str_eta = "eta_"+str(int_etamin)+"_"+str(int_etamax)
                dir = top_dir+"pid_"+str(pdgid)+"_"+str_eta
                jobname = "runParam_"+"pid_"+str(pdgid)+"_"+str_eta
                run = jobname
                script_exe = run + ".sh"
                script_sub = "run.sub"
                script_runParamEtaSlice = "runParamEtaSlice.C"
                if not os.path.exists(dir):
                    os.makedirs(dir)
                os.chdir(dir)
                os.system("echo ")
                os.system("echo \"Current directory :\" $PWD ")
                # cp exe_temp and modify
                os.system("cp ..\/..\/"+exe_temp+" "+script_exe)
                cwd = os.getcwd()
                with open(script_exe, 'r') as f:
                    text = f.read()
                text = text.replace('@SUBMIT_DIR@', cwd)
                with open(script_exe, 'w') as f:
                    f.write(text)
                os.system("chmod +x "+script_exe)
                # cp sub_temp and modify
                os.system("cp ..\/..\/"+sub_temp+" "+script_sub)
                with open(script_sub, 'r') as f:
                    text = f.read()
                text = text.replace('@EXE@', script_exe)
                text = text.replace('@RUN@', run)
                text = text.replace('@MEM@', str(memory))
                text = text.replace('@MAXRUNTIME@', str(MaxRuntime))
                text = text.replace('@EMAIL@', email)
                with open(script_sub, 'w') as f:
                    f.write(text)
                # cp runEpara_temp and modify
                os.system("cp ..\/..\/"+runParamEtaSlice_temp+" "+script_runParamEtaSlice)
                with open(script_runParamEtaSlice, 'r') as f:
                    text = f.read()
                text = text.replace('@PID@', str(pdgid))
                text = text.replace('@EMIN@', str(Emin))
                text = text.replace('@EMAX@', str(Emax))
                text = text.replace('@ETAMIN@', str(eta))
                text = text.replace('@DIR@', '\\"'+outDir+'\\"')
                with open(script_runParamEtaSlice, 'w') as f:
                    f.write(text)
                # cp init_file and run_ParamEtaSlice
                os.system("cp ..\/..\/"+init_file+" "+init_file)
                os.system("cp ..\/..\/"+run_ParamEtaSlice+" "+run_ParamEtaSlice)
                os.system("cp ..\/..\/"+run_CreateParam+" "+run_CreateParam)
                os.system("cp ..\/..\/"+database+" "+database)



                # submit script_sub
                CondorSubmitCommand = "condor_submit "+script_sub
                print (CondorSubmitCommand)
                os.system(CondorSubmitCommand)

                os.chdir("../../")
                os.system("echo \"Current directory :\" $PWD ")


    if options.checkOnly:
        print("Checking condor jobs...")
        failedJobs=[]
        failedJobs = checkParamEtaSlices(outDir)
        pprint(failedJobs)

        if options.writeFile:
            failed_dsid = failedJobs.keys()
            print("writing the dsids in a txt file...")
            with open(options.writeFile, 'w') as outFile:
                outFile.write('\n'.join(failed_dsid))


if __name__ == '__main__':
    main()




