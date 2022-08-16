import tensorflow as tf 
import numpy as np
import pandas as pd
import glob 


class DataLoader:

    def __init__(self, config):

        self.config = config

        self.n_tau = self.config["Setup"]["n_tau"] # number of taus/batch
        self.n_batches = self.config["Setup"]["n_batches"] # number of batches for training
        self.n_batches_val = self.config["Setup"]["n_batches_val"] # number of batches for validation
        self.n_epochs = self.config["Setup"]["n_epochs"] # number of epochs
        self.epoch = self.config["Setup"]["epoch"] # starting epoch
        self.val_split = self.config["Setup"]["val_split"] 
        self.dropout_rate = self.config["Setup"]["dropout"]
        self.activation = self.config["Setup"]["activation"]
        self.optimiser = self.config["Setup"]["optimiser"]
        self.learning_rate = self.config["Setup"]["learning_rate"]
        self.model_name = self.config["Setup"]["model_name"]
        self.file_path = self.config["Setup"]["input_dir"]
        print(self.file_path)
        files = glob.glob(self.file_path + "/*.pkl")
        self.train_files, self.val_files = np.split(files, [int(len(files)*(1-self.val_split))])
        print("Files for training:", len(self.train_files))
        # print(self.train_files)
        print("Files for validation:", len(self.val_files))
        # print(self.val_files)

    def get_generator(self, primary_set = True, show_progress = False, evaluation = False):

        _files = self.train_files if primary_set else self.val_files
        print(("Training" if primary_set else "Validation") + " file list loaded" )
        if len(_files)==0:
            raise RuntimeError(("Training" if primary_set else "Validation")+\
                                " file list is empty.")

        n_batches = self.n_batches if primary_set else self.n_batches_val

        def _generator():
            counter = 0
            while counter<n_batches:
                for j in range(len(_files)):
                    df = pd.read_pickle(_files[j])
                    for i in range(len(df)):
                        Tracks = df["Tracks"][i]
                        ECAL = df["ECAL"][i]
                        PF_HCAL = df["PF_HCAL"][i]
                        PF_ECAL = df["PF_ECAL"][i]
                        addTracks = df["addTracks"][i]
                        x = (np.stack([Tracks, ECAL, PF_HCAL, PF_ECAL, addTracks], axis=-1))
                        DM = df["DM"][i]
                        if evaluation:
                            y = DM
                        else:
                            if DM == 0 or DM == 10:
                                y = tf.one_hot(0, 3) # no pi0
                            elif DM ==1 or DM ==11:
                                y = tf.one_hot(1, 3) # one pi0
                            elif DM == 2:
                                y = tf.one_hot(2, 3) # two pi0
                            else: 
                                raise RuntimeError(f"Unknown DM {DM}")
                        counter += 1
                        yield (x,y)

        return _generator

