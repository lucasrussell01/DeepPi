#!/usr/bin/env python

import sys
import os
import string
import shlex
from subprocess import Popen, PIPE
import argparse
import numpy as np

parser = argparse.ArgumentParser(description='Generate images from RHTree ROOT files')
parser.add_argument('--sample', required=True, type=str, help="Sample to submit to batch")
parser.add_argument('--path_to_list', required=True, type=str, help="Path to file lists")
parser.add_argument('--d_cache', required=True, type=str, help="path to crab output in dcache")
parser.add_argument('--run_name', required=True, type=str, help="Run name specified when creating file list")
parser.add_argument('--save_path', required=True, type=str, help="Where to store jobs and files")

args = parser.parse_args()

dcache = args.d_cache # path to crab outputs in dcache (filelists don't include these)

current_dir = os.getcwd()

def CreateBatchJob(name, cmd_list, current_directory):
  if os.path.exists(name): os.system('rm %(name)s' % vars())
  os.system('echo "#!/bin/bash" >> %(name)s' % vars())
  os.system('echo "source /vols/grid/cms/setup.sh" >> %(name)s' % vars())
  os.system('echo "cd %(current_directory)s" >> %(name)s' % vars())
  os.system('echo "export X509_USER_PROXY=$HOME/cms.proxy">> %(name)s' % vars())
  os.system('echo "cmsenv" >> %(name)s' % vars())
  for cmd in cmd_list:
    os.system('echo "%(cmd)s" >> %(name)s' % vars())
  os.system('chmod +x %(name)s' % vars())
  print "Created job:",name

def SubmitBatchJob(name,time=180,memory=24,cores=1):
  error_log = name.replace('.sh','_error.log')
  output_log = name.replace('.sh','_output.log')
  if os.path.exists(error_log): os.system('rm %(error_log)s' % vars())
  if os.path.exists(output_log): os.system('rm %(output_log)s' % vars())
  os.system('qsub -e %(error_log)s -o %(output_log)s -V -q hep.q -l h_vmem=24G -cwd %(name)s' % vars())



sample = args.sample
if sample == "GluGluHToTauTau_M125":
  alias = "ggHTT_powheg_"
elif sample == "GluGluHToTauTau_M-125":
  alias = "ggHTT_madgraph_"
elif sample == "WminusHToTauTau_M125":
  alias = "WminusHTT_"
elif sample == "WplusHToTauTau_M125":
  alias = "WplusHTT_"
elif sample == "VBFHToTauTau_M125":
  alias = "VBFHTT_"
else:
  alias = "unkwn"
  raise Exception("Unknown sample")

path_to_filelist = args.path_to_list + "/" + args.run_name + "_MC_106X_" + sample + ".dat"

with open(path_to_filelist) as f:
  file_list = [line.rstrip('\n') for line in f.readlines()]
  n_files = len(file_list)
  print("File list loaded successfully, %(n_files)s to process" % vars())


storage = args.save_path
if not os.path.isdir(storage):os.system("mkdir %(storage)s" % vars())
output = storage + args.sample 




n = 0

for f in file_list:
  filename = alias + str(n)  
  file = dcache + f 
  if not os.path.isdir(output):os.system("mkdir %(output)s" % vars())
  if not os.path.isdir('%(output)s/jobs' % vars()):os.system("mkdir %(output)s/jobs" % vars())
  if not os.path.isdir('%(output)s/workspaces' % vars()):os.system("mkdir %(output)s/workspaces" % vars())

  run_cmd = "python GenerateImagesFile.py --file=%(file)s --alias=%(filename)s --save_path=%(output)s" % vars()

  job_file = "%(output)s/jobs/%(filename)s.sh" % vars()
  CreateBatchJob(job_file, [run_cmd], current_dir)
  SubmitBatchJob(job_file)
  n+=1