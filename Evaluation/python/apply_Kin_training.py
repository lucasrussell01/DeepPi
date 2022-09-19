import tensorflow as tf
from tensorflow.keras.models import load_model
import json
import yaml
import os
import numpy as np
from tqdm import tqdm
import pandas as pd
import sys
sys.path.append("../../Training/python")
from DataLoader import DataLoader
os.environ["CUDA_VISIBLE_DEVICES"]="-1" # don't need to use GPU
import argparse

parser = argparse.ArgumentParser(description='Generate Kinematic predictions for DeepPi training')
parser.add_argument('--expID', required=True, type=str, help="Experiment ID")
parser.add_argument('--runID', required=True, type=str, help="Run ID")
parser.add_argument('--n_tau', required=False, default = 50000, type=int, help="n_tau")


args = parser.parse_args()


path_to_mlflow = "../../Training/python/mlruns/"
expID = args.expID 
runID = args.runID

path_to_artifacts = path_to_mlflow + expID + "/" + runID + "/artifacts"

def pos_xyz(eta, phi): # get xyz from 0,0,0 
    rho = 129 # distance from beamline to ECAL entrance
    theta = 2*np.arctan(np.exp(-eta))
    z = rho/np.tan(theta)
    x = rho*np.cos(phi)
    y = rho*np.sin(phi)
    return np.array([x, y, z])

def pi0_xyz(pos_arr, v_arr): # get xyz of pi0 from the PV
    pi0 = pos_arr - v_arr
    return pi0

def get_pi0_eta(pos_xyz):
    theta = np.arctan2(129, pos_xyz[2])
    eta = -np.log(np.tan(theta/2))
    return eta

def get_pi0_phi(pos_xyz):
    phi = np.arctan2(pos_xyz[1], pos_xyz[0])
    return phi

def test(x, model):
    y_pred = model(x, training=False)[0] # take kinematic output
    return np.array(y_pred)

def get_HPS_eta(p_vect):
    p = np.sqrt(p_vect[0]**2 + p_vect[1]**2 + p_vect[2]**2)
    theta = np.arccos(p_vect[2]/p) # pz/p
    eta = -np.log(np.tan(theta/2))
    return eta

def get_HPS_phi(p_vect):
    phi = np.arctan2(p_vect[1], p_vect[0])
    return phi

def get_HPS_p(p_vect):
    p = np.sqrt(p_vect[0]**2 + p_vect[1]**2 + p_vect[2]**2)
    return p

# Load training cfg
with open(f'{path_to_artifacts}/input_cfg/training_cfg.yaml') as file:
    training_cfg = yaml.full_load(file)
    print("Training Config Loaded")
training_cfg["Setup"]["input_dir"] = '/vols/cms/lcr119/Images/Images_MVA/Evaluation'
training_cfg["Setup"]["n_batches"] = args.n_tau
training_cfg["Setup"]["n_batches_val"] = 0
training_cfg["Setup"]["val_split"] = 0

print(training_cfg)

# Load evaluation dataset
dataloader = DataLoader(training_cfg)
gen_eval = dataloader.get_generator_v2(primary_set = True, evaluation=True)


if training_cfg["Setup"]["HPS_features"]:
    print("Warning: Model was trained with HPS vars")
    input_shape = (((33, 33, 5), 31), None, 3, None, None, 3,  2)
    input_types = ((tf.float32, tf.float32), tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32)
else:
    input_shape = ((33, 33, 5), None, 3, None, None, 3,  2)
    input_types = (tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32)


data_eval = tf.data.Dataset.from_generator(
    gen_eval, output_types = input_types, output_shapes = input_shape
    ).prefetch(tf.data.AUTOTUNE).batch(1).take(dataloader.n_batches)
print("Dataset Loaded")

# Load model
with open(f'{path_to_artifacts}/input_cfg/metric_names.json') as f:
    metric_names = json.load(f)
    path_to_model = f'{path_to_artifacts}/model'

model = load_model(path_to_model, {name: lambda _: None for name in metric_names.keys()})
print("Model loaded")


pbar = tqdm(total = dataloader.n_batches)

rel_p = []
rel_p_pred = []
rel_eta = []
rel_eta_pred = []
rel_phi = []
rel_phi_pred = []
pi0_eta = []
pi0_phi = []
pi0_eta_pred = []
pi0_phi_pred = []
pi0_eta_HPS = []
pi0_phi_HPS = []
pi0_p_HPS = []

# Generate predictions
i = 0
for elem in data_eval:
    x, y, PV, DM, HPSDM, HPS_pi0, jetpos = elem
    HPS_pi0 = np.array(HPS_pi0)[0]
    jetpos = np.array(jetpos[0])
    if DM != 1 and DM != 11: # only kinematic regress DM 1 for now
        i+=1
        if i%10 ==0:
            pbar.update(10)
        continue
    elif HPSDM == 1 or HPSDM == 11: # make sure HPS has reco tau as same DM? 
        y_pred = test(x, model)
        y = np.array(y)[0]
        rel_p.append(y[0])
        rel_eta.append(y[1])
        rel_phi.append(y[2])
        rel_p_pred.append(y_pred[0])
        rel_eta_pred.append(y_pred[1])
        rel_phi_pred.append(y_pred[2])
        eta_orig = jetpos[0] + y[1]
        eta_orig_pred = jetpos[0] + y_pred[1]
        phi_orig = jetpos[1] + y[2]
        phi_orig_pred = jetpos[1] + y_pred[2]
        pos = pos_xyz(eta_orig, phi_orig)
        pos_pred = pos_xyz(eta_orig_pred, phi_orig_pred)
        pi0 = pi0_xyz(pos, np.array(PV))[0]
        pi0_pred = pi0_xyz(pos_pred, np.array(PV))[0]
        pi0_eta.append(get_pi0_eta(pi0))
        pi0_phi.append(get_pi0_phi(pi0))
        pi0_eta_pred.append(get_pi0_eta(pi0_pred))
        pi0_phi_pred.append(get_pi0_phi(pi0_pred))
        pi0_eta_HPS.append(get_HPS_eta(HPS_pi0))
        pi0_phi_HPS.append(get_HPS_phi(HPS_pi0))
        pi0_p_HPS.append(get_HPS_p(HPS_pi0))
        i+=1
        if i%10 ==0:
            pbar.update(10)
    else:
        i+=1
        if i%10 ==0:
            pbar.update(10)
        continue

print(f"Total of {len(rel_p)} DM 1 or 11 taus evaluated")


df = pd.DataFrame()
df["relp"] = rel_p 
df["relp_pred"] = rel_p_pred
df["releta"] = rel_eta
df["releta_pred"] = rel_eta_pred
df["relphi"] = rel_phi
df["relphi_pred"] = rel_phi_pred
df["pi0_phi"] = pi0_phi
df["pi0_eta"] = pi0_eta
df["pi0_phi_pred"] = pi0_phi_pred
df["pi0_eta_pred"] = pi0_eta_pred
df["pi0_eta_HPS"] = pi0_eta_HPS
df["pi0_phi_HPS"] = pi0_phi_HPS
df["pi0_p_HPS"] = pi0_p_HPS


print(df)

print("Predictions computed")

save_folder = path_to_artifacts + "/predictions"
if not os.path.exists(save_folder):
    os.makedirs(save_folder)
savepath = save_folder + "/kinematic_pred_ggH.pkl"

df.to_pickle(savepath)

print("Predictions saved in artifacts/predictions")

