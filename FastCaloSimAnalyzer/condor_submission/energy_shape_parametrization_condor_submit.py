#!/usr/bin/evn python
from __future__ import print_function
import sys
import os
import glob
import fileinput
import socket
import commands
os.system("""echo "TFile.Recover 0" >> .rootrc""")
import ROOT
from argparse import ArgumentParser
from pprint import pprint
#--------------------------------------------------
# user information
email = "a.hasib@cern.ch"

# set general parameters
memory = 2000
# flavor = "tomorrow"
MaxRuntime = 5*60*60

metadata = "inputSampleList.txt"
outDir = "/eos/atlas/user/a/ahasib/public/Simul-FastCalo/ParametrizationProductionVer07/"
plotDir = "/eos/user/a/ahasib/www/Simul-FastCalo/ParametrizationProductionVer07/"
version = "ver07"

runEpara_temp = "runEpara_temp.C"
runMeanRZ_temp = "runMeanRZ_temp.C"
runShape_temp = "runShape_temp.C"
sub_temp = "template_submit.sub"
exe_temp = "template_run.sh"


# energy parametrization related parameters
npca_primary = 5
npca_secondary = 1
do_validation = 1

init_epara = "initTFCSAnalyzer.C"
run_epara = "run_epara.cxx"

# shape parametrization related parameters
energy_cutoff = 0.9995

dsid_zv0 = -999
do_phiSymm = 1
do_ZvertexStudies = 0


init_shape = "initTFCSAnalyzer.C"
run_shape  = "runTFCS2DParametrizationHistogram.cxx"

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

def checkParametrizations(destination, inFile):
    failed_jobs = {}
    
    if inFile:
      with open(inFile) as DSIDList:
        for DSID in DSIDList:
          status = []  
          DSID = DSID.strip("\n")
          shapefile = "mc16_13TeV."+DSID+".*.shapepara.*.root"
          energyfile = "mc16_13TeV."+DSID+".*.secondPCA.*.root"
          extrapolfile = "mc16_13TeV."+DSID+".*.extrapol.*.root"

          shapefileNames = commands.getoutput("ls {destination}/{dataset}".format(destination=destination,dataset=shapefile))
          energyfileNames = commands.getoutput("ls {destination}/{dataset}".format(destination=destination,dataset=energyfile))
          extrapolfileNames = commands.getoutput("ls {destination}/{dataset}".format(destination=destination,dataset=extrapolfile))
          
          
          print("Checking: ", shapefileNames, energyfileNames, extrapolfileNames, sep="\n")
          print("=====================================================================")
          if "cannot" in shapefileNames or "cannot" in energyfileNames or "cannot" in extrapolfileNames:
            if "cannot" in shapefileNames:
                status.append("shape file doesn't exist")
            if "cannot" in energyfileNames:
                status.append("energy file doesn't exist")
            if "cannot" in extrapolfileNames:
                status.append("extrapol file doesn't exist")
          else:
            fshape = ROOT.TFile(shapefileNames)
            if fshape.IsZombie():
                status.append("shape file corrupted")
                fshape.Close()
            fenergy = ROOT.TFile(energyfileNames)
            if fenergy.IsZombie():
                status.append("energy file corrupted")
                fenergy.Close()
            fextrapol = ROOT.TFile(extrapolfileNames)
            if fextrapol.IsZombie():
                status.append("extrapol file corrupted")
                fextrapol.Close() 
            if status:
                failed_jobs[DSID] = status            
    return failed_jobs



def main():

    options = getArguments()

    # check to see if running on LXPLUS
    try:
        host = socket.gethostname()
        if not 'lxplus' in host  and not options.checkOnly:
            raise Exception
    except Exception:
        print("Condor jobs have to be submitted from a LXPLUS node")
        sys.exit(1)

    # check if a list of dsids is provided
    if not options.input:
        print("You need to provide a list of DSIDs you want to run on")
        sys.exit(1)

        # for each input dsid lists
    if not options.checkOnly:
        print("Running Condor Submission ...")

        for inFile in options.input:
            dsids = getListOfDSIDs(inFile)
            # for each dsid copy, modify all files and submit
            for dsid in dsids:
                dir = "run/dsid_" + str(dsid)
                jobname = "runDSID"+"_"+str(dsid)
                run = jobname
                script_exe = run + ".sh"
                script_sub = "run.sub"
                script_runEpara = "runEpara.C"
                script_runShape = "runShape.C"
                script_runMeanRZ = "runMeanRZ.C"

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
                os.system("cp ..\/..\/"+runEpara_temp+" "+script_runEpara)
                with open(script_runEpara, 'r') as f:
                    text = f.read()
                text = text.replace('@DSID@', str(dsid))
                text = text.replace('@METADATA@', '\\"'+metadata+'\\"')
                text = text.replace('@DIR@', '\\"'+outDir+'\\"')
                text = text.replace('@NPCA1@', str(npca_primary))
                text = text.replace('@NPCA2@', str(npca_secondary))
                text = text.replace('@VALIDATION@', str(do_validation))
                text = text.replace('@VER@', '\\"'+version+'\\"')
                text = text.replace('@PLOTDIR@', '\\"'+plotDir+'\\"')
                with open(script_runEpara, 'w') as f:
                    f.write(text)
                 # cp runMeanRZ_temp and modify
                os.system("cp ..\/..\/"+runMeanRZ_temp+" "+script_runMeanRZ)
                with open(script_runMeanRZ, 'r') as f:
                    text = f.read()
                text = text.replace('@DSID@', str(dsid))
                text = text.replace('@DSIDZV0@', str(dsid_zv0))
                text = text.replace('@METADATA@', '\\"'+metadata+'\\"')
                text = text.replace('@DIR@', '\\"'+outDir+'\\"')
                text = text.replace('@VER@', '\\"'+version+'\\"')
                text = text.replace('@CUTOFF@', str(energy_cutoff))
                text = text.replace('@PLOTDIR@', '\\"'+plotDir+'\\"')
                text = text.replace('@DO2DPARAM@', str(0))
                text = text.replace('@PHISYMM@', str(do_phiSymm))
                text = text.replace('@DOMEANRZ@', str(1))
                text = text.replace('@USEMEANRZ@', str(0))
                text = text.replace('@DOZVERTEXSTUDIES@', str(do_ZvertexStudies))

                with open(script_runMeanRZ, 'w') as f:
                    f.write(text)

                # cp runShape_temp and modify
                os.system("cp ..\/..\/"+runShape_temp+" "+script_runShape)
                with open(script_runShape, 'r') as f:
                    text = f.read()
                text = text.replace('@DSID@', str(dsid))
                text = text.replace('@DSIDZV0@', str(dsid_zv0))
                text = text.replace('@METADATA@', '\\"'+metadata+'\\"')
                text = text.replace('@DIR@', '\\"'+outDir+'\\"')
                text = text.replace('@VER@', '\\"'+version+'\\"')
                text = text.replace('@CUTOFF@', str(energy_cutoff))
                text = text.replace('@PLOTDIR@', '\\"'+plotDir+'\\"')
                text = text.replace('@DO2DPARAM@', str(1))
                text = text.replace('@PHISYMM@', str(do_phiSymm))
                text = text.replace('@DOMEANRZ@', str(0))
                text = text.replace('@USEMEANRZ@', str(1))
                text = text.replace('@DOZVERTEXSTUDIES@', str(do_ZvertexStudies))



                with open(script_runShape, 'w') as f:
                    f.write(text)
                # cp init_epara and run_epara
                os.system("cp ..\/..\/"+init_epara+" "+init_epara)
                os.system("cp ..\/..\/"+run_epara+" "+run_epara)
                # cp init_shape and run_shape
                os.system("cp ..\/..\/"+init_shape+" "+init_shape)
                os.system("cp ..\/..\/"+run_shape+" "+run_shape)
                # cp metadata file
                os.system("cp ..\/..\/"+metadata+" "+metadata)

                # submit script_sub
                CondorSubmitCommand = "condor_submit "+script_sub
                print (CondorSubmitCommand)
                os.system(CondorSubmitCommand)

                os.chdir("../../")
                os.system("echo \"Current directory :\" $PWD ")


    if options.checkOnly:
        print("Checking condor jobs...")
        failedJobs=[]
        for inFile in options.input:
            failedJobs = checkParametrizations(outDir, inFile)
            pprint(failedJobs)

        if options.writeFile:
            failed_dsid = failedJobs.keys()
            print("writing the dsids in a txt file...")
            with open(options.writeFile, 'w') as outFile:
                outFile.write('\n'.join(failed_dsid))


if __name__ == '__main__':
    main()




