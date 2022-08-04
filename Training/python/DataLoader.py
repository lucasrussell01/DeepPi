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
            
            if show_progress and n_batches>0:
                pbar = tqdm(total = n_batches)
            
            df = pd.read_pickle(_files[0])[:1000]
            # print(len(df["DM"])/n_tau)
            for i in range(len(df)): # try not batching 
                x = (df["Tracks"][i], df["ECAL"][i], df["HCAL"][i])
                print(type(x))
                y = (df["DM"][i])
                yield(x,y)
   
        return _generator

    # def gen_test(self):

    #     _files = self.train_files 
    #     df = pd.read_pickle(_files[0])
    #     df = pd.read_pickle(_files[0])
    #     # print(len(df["DM"])/n_tau)
    #     for i in range(len(df)): # try not batching 
    #         x = (df["Tracks"][i], df["ECAL"][i], df["HCAL"][i])
    #         print(type(x))
    #         y = (df["DM"][i])
    #         break


        # for batch in np.array_split(df, len(df["DM"])/n_tau): # divide into batches

source_dir = "/home/hep/lcr119/Datasets/"
sample = "GluGluHToTauTau_M125"
n_tau = 100
n_batches = 200
load = DataLoader(source_dir, sample, 100, 200, 0)

#separte for testing
gen = load.get_generator()

print("Generator created")
ds = tf.data.Dataset.from_generator(gen)
print("TF dataset")


# print(load.get_generator())
# print(files)
