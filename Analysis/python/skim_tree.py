import argparse
import os
import ROOT

ROOT.gROOT.SetBatch(True)
n_threads = 4 # multithread
ROOT.ROOT.EnableImplicitMT(4)