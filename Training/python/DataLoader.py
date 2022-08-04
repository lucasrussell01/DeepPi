import tensorflow as tf 
import numpy as np
import pandas as pd
import glob 


class DataLoader:

    def __init__(self, source_dir, sample, n_batches, n_tau, n_batches_val):

        self.n_batches = n_batches # number of batches for training
        self.n_batches_val = n_batches_val # number of batches for validation
        self.n_tau = n_tau # number of taus/batch
        self.file_path = source_dir + sample
        self.files = glob.glob(self.file_path + "/*.pkl")
        self.train_files = self.files # add validation files later
        self.val_files = []

    def get_generator(self, primary_set = True, return_truth = True, show_progress = False):

        _files = self.train_files if primary_set else self.val_files
        print(("Training" if primary_set else "Validation") + " file list loaded" )
        if len(_files)==0:
            raise RuntimeError(("Training" if primary_set else "Validation")+\
                                " file list is empty.")

        n_batches = self.n_batches if primary_set else self.n_batches_val

        def _generator():
            df = pd.read_pickle(_files[0])
            for i in range(len(df)):
                Tracks = df["Tracks"][i]
                ECAL = df["ECAL"][i]
                HCAL = df["HCAL"][i]
                x = (Tracks, ECAL, HCAL)
                DM = df["DM"][i]
                if DM == 0 or DM == 10:
                    y = 0 # no pi0
                elif DM ==1 or DM ==1:
                    y = 1 # one pi0
                elif DM == 2:
                    y = 2 # two pi0
                yield (x,y)
   
        return _generator

