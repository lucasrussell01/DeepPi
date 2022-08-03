# import pandas as pd
import numpy as np
import ROOT as R
import os
import glob
# import tensorflow as tf

file_path = "root://gfe02.grid.hep.ph.ic.ac.uk:1097/store/user/lrussell/DetectorImages_MC_106X_2018/GluGluHToTauTau_M125_TuneCP5_13TeV-powheg-pythia8/crab_GluGluHToTauTau_M125/220802_171452/0000"

print(file_path + '/*.root')

data_files = glob.glob(file_path + '/*.root') 
# print(data_files)

test_file = "root://gfe02.grid.hep.ph.ic.ac.uk:1097/store/user/lrussell/DetectorImages_MC_106X_2018/GluGluHToTauTau_M125_TuneCP5_13TeV-powheg-pythia8/crab_GluGluHToTauTau_M125/220802_171452/0000/EventTree_2.root"

rhTree = R.TChain("recHitAnalyzer/RHTree")
rhTree.Add(test_file)
nEvts = rhTree.GetEntries()
print(nEvts)

branches = list(rhTree.GetListOfBranches())

# for b in branches:
#     print(b)





