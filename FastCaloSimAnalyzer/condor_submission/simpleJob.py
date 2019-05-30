#!/bin/env python                                                                                                                                            

import os
import sys
import commands

def submitJob(batchDir, memory, email, jobName, command, parameters):
    
    baseDir   = os.getenv("PWD")    
    #here will go the logs from submitted jobs
    logsDir = os.path.join(batchDir,"logs")
    rc,out = commands.getstatusoutput("mkdir -p " + logsDir)
    params=" ".join(parameters)
    print "Basedir: " + baseDir
    print "Making condor job with command " + command + " " + params

    template = """

universe                = vanilla
executable              = %(baseDir)s/condor_submission/simpleJob.sh 
arguments               = %(baseDir)s %(command)s %(params)s
output                  = %(logsDir)s/%(jobName)s.$(ClusterId).$(ProcId).out
error                   = %(logsDir)s/%(jobName)s.$(ClusterId).$(ProcId).err
log                     = %(logsDir)s/%(jobName)s.$(ClusterId).log
request_cpus            = 1
request_memory          = %(memory)s
+MaxRuntime             = 36*60*60 
environment = \"HOME=$ENV(HOME)\"
environment = \"USER=$ENV(USER)\"
transfer_executable = True
notification            = Complete
notify_user             = %(email)s

queue 1

""" % { 'baseDir'     : baseDir,
	'memory'      : memory,
	'email'       : email,
	'logsDir'     : logsDir,
	'jobName'     : jobName,
	'command'     : command,
	'params'      : params
	}

    #create submission file for batch job submission
    
    submitFileName = batchDir + '/run_%s.sub'%(jobName)
    submitFile = open(submitFileName,'w')
    submitFile.write(template)
    submitFile.close()

    submCmd="condor_submit %s" % (submitFileName)  
    #print submCmd
    rc,out = commands.getstatusoutput(submCmd)
    print "task submitted, rc= ",rc
    print "submission output: ",out
    

if __name__=="__main__":


    batchDir = sys.argv[1]
    memory = sys.argv[2]
    email = sys.argv[3]
    jobName = sys.argv[4]
    command = sys.argv[5]
    parameters = sys.argv[6:len(sys.argv)]
    
    submitJob(batchDir, memory, email, jobName, command, parameters)
