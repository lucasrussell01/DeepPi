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
        self.regress_kinematic = self.config["Setup"]["kinematic"]
        self.kDM = self.config["Setup"]["DM_importance"]
        self.kKin = self.config["Setup"]["kin_importance"]
        self.use_HPS = self.config["Setup"]["HPS_features"]
        self.use_weights = self.config["Setup"]["use_weights"]
        self.file_path = self.config["Setup"]["input_dir"]
        print(self.file_path)
        files = glob.glob(self.file_path + "/*.pkl")
        self.train_files, self.val_files = np.split(files, [int(len(files)*(1-self.val_split))])
        print("Files for training:", len(self.train_files))
        # print(self.train_files)
        print("Files for validation:", len(self.val_files))
        # print(self.val_files)

    def get_generator_v1(self, primary_set = True, show_progress = False, DM_evaluation = False, Kin_evaluation = False):

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
                        # Image inputs
                        Tracks = df["Tracks"][i]
                        ECAL = df["ECAL"][i]
                        PF_HCAL = df["PF_HCAL"][i]
                        PF_ECAL = df["PF_ECAL"][i]
                        addTracks = df["addTracks"][i]
                        # Clip outliers to max determined from 99.7%:
                        if np.sum(Tracks)>200:
                            Tracks = 200*Tracks/np.sum(Tracks)
                        if np.sum(ECAL)>200:
                            ECAL = 200*ECAL/np.sum(ECAL)
                        if np.sum(PF_HCAL)>150:
                            PF_HCAL = 150*PF_HCAL/np.sum(PF_HCAL)
                        if np.sum(PF_ECAL)>200:
                            PF_ECAL = 200*PF_ECAL/np.sum(PF_ECAL)
                        if np.sum(addTracks)>70:
                            addTracks = 70*addTracks/np.sum(addTracks)
                        x = (np.stack([Tracks, ECAL, PF_HCAL, PF_ECAL, addTracks], axis=-1))
                        if self.use_HPS:
                            x_mass = (np.stack([df["tau_dm"][i], df["tau_pt"][i], df["tau_E"][i], df["tau_eta"][i], df["tau_mass"][i],
                            df["pi0_dEta"][i], df["pi0_dPhi"][i], df["strip_mass"][i], df["strip_pt"][i], 
                            df["rho_mass"][i], df["mass0"][i], df["mass1"][i], df["mass2"][i]], axis=-1))
                            x = tuple([x, x_mass])
                        DM = df["DM"][i]
                        if self.regress_kinematic:
                            max_index = np.where(df["relp"][i] == np.max(df["relp"][i]))[0] # find leading neutral
                            yKin = (df["relp"][i][max_index][0], df["releta"][i][max_index][0], df["relphi"][i][max_index][0])
                        if DM_evaluation:
                            yDM = DM
                            HPSDM = df["tau_dm"][i]
                            yield(x, yDM, HPSDM)
                        elif Kin_evaluation:
                            PV = df["PV"][i]
                            HPSDM = df["tau_dm"][i]
                            HPS_pi0 = [df["pi0_px"][i], df["pi0_py"][i], df["pi0_pz"][i]]
                            jet = [df["jet_eta"][i], df["jet_phi"][i]]
                            yield(x, yKin, PV, DM, HPSDM, HPS_pi0, jet)
                        else:
                            if DM == 0 or DM == 10:
                                yDM = tf.one_hot(0, 3) # no pi0
                                w = 0
                            elif DM ==1 or DM ==11:
                                yDM = tf.one_hot(1, 3) # one pi0
                                w = 1
                            elif DM == 2:
                                yDM = tf.one_hot(2, 3) # two pi0
                                w = 0
                            else: 
                                raise RuntimeError(f"Unknown DM {DM}")
                            if self.regress_kinematic:
                                yield (x, yDM, yKin, w)
                            else:
                                yield (x, yDM)
                        counter += 1

        return _generator

    def get_generator_v2(self, primary_set = True, show_progress = False, evaluation = False):

        _files = self.train_files if primary_set else self.val_files
        print(("Training" if primary_set else "Validation") + " file list loaded" )
        if len(_files)==0:
            raise RuntimeError(("Training" if primary_set else "Validation")+\
                                " file list is empty.")

        n_batches = self.n_batches if primary_set else self.n_batches_val

        def _generator():
            for j in range(len(_files)):
                df = pd.read_pickle(_files[j])
                for i in range(len(df)):
                    DM = df["DM"][i]
                    if self.use_HPS:
                        HPSDM = df["tau_dm"][i] # use HPS DM for cuts
                    if DM != 1 and DM != 11: # v2 train only on 1 pi0 taus for now
                        continue
                    # Image inputs
                    Tracks = df["Tracks"][i]
                    ECAL = df["ECAL"][i]
                    PF_HCAL = df["PF_HCAL"][i]
                    PF_ECAL = df["PF_ECAL"][i]
                    addTracks = df["addTracks"][i]
                    # Clip outliers to max determined from 99.7%:
                    if np.sum(Tracks)>200:
                        Tracks = 200*Tracks/np.sum(Tracks)
                    if np.sum(ECAL)>200:
                        ECAL = 200*ECAL/np.sum(ECAL)
                    if np.sum(PF_HCAL)>150:
                        PF_HCAL = 150*PF_HCAL/np.sum(PF_HCAL)
                    if np.sum(PF_ECAL)>200:
                        PF_ECAL = 200*PF_ECAL/np.sum(PF_ECAL)
                    if np.sum(addTracks)>70:
                        addTracks = 70*addTracks/np.sum(addTracks)
                    x = (np.stack([Tracks, ECAL, PF_HCAL, PF_ECAL, addTracks], axis=-1))
                    if self.use_HPS:
                        x_mass = (np.stack([df["tau_dm"][i], df["tau_pt"][i], df["tau_E"][i], df["tau_eta"][i], df["tau_mass"][i],
                        df["pi0_dEta"][i], df["pi0_dPhi"][i], df["strip_mass"][i], df["strip_pt"][i], 
                        df["rho_mass"][i], df["mass0"][i], df["mass1"][i], df["mass2"][i], df["pi0_E"][i]], axis=-1))
                        x = tuple([x, x_mass])
                    yp = df["relp"][i][0] # pi0 momentum
                    if evaluation:
                        PV = df["PV"][i]
                        HPSDM = df["tau_dm"][i]
                        HPS_pi0 = [df["pi0_px"][i], df["pi0_py"][i], df["pi0_pz"][i]]
                        jet = [df["jet_eta"][i], df["jet_phi"][i]]
                        yield(x, yp, PV, DM, HPSDM, HPS_pi0, jet)
                    elif self.use_weights:
                        weights = np.array([1.35282914, 0.42862416, 0.40479484, 0.45116552, 0.54597761, 0.66752737, 0.8515014,
                                         1.10876879, 1.42314426, 1.74725353, 2.24585584, 2.80869932, 3.62901786, 4.80816246, 
                                         6.44931241, 8.32889344, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
                        weight_bins = np.arange(0, 155, 5)
                        weight_bins[-1] = 1000 # increase to cover up to 1TeV
                        if yp < 1000:
                            i_weight = np.digitize(yp, weight_bins) # find bin in spectrum
                            w = weights[i_weight-1] # find corresponding weight
                        else:
                            print(f"Warning: momentum value: {yp} out of range, setting weight to 0")
                            w = 0.0
                        yield(x, yp, w)
                    else:
                        yield(x, yp)

        return _generator
