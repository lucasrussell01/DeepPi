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
training_cfg["Setup"]["input_dir"] = '/vols/cms/lcr119/Images/DeepPi_v1/EGammaCentering/Evaluation'
training_cfg["Setup"]["n_batches"] = args.n_tau
training_cfg["Setup"]["n_batches_val"] = 0
training_cfg["Setup"]["val_split"] = 0

print(training_cfg)

# Load evaluation dataset
dataloader = DataLoader(training_cfg)
gen_eval = dataloader.get_generator_v2(primary_set = True, evaluation=True)


if training_cfg["Setup"]["HPS_features"]:
    print("Warning: Model was trained with HPS vars")
    input_shape = (((33, 33, 5), 31), None, 3, None, None, 5,  2)
    input_types = ((tf.float32, tf.float32), tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32)
else:
    input_shape = ((33, 33, 5), None, 3, None, None, 5,  2)
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

genDM = []
# Gen information:
rel_p = [] 
rel_eta = []
rel_phi = []
# Predictions
rel_p_pred = []
rel_eta_pred = []
rel_phi_pred = []
# Store HPS DM for comparisons
HPSDM_list = [] 
# True Pi0 Position
pi0_eta = []
pi0_phi = []
# CNN predicted pi0 position
pi0_eta_pred = []
pi0_phi_pred = []
# HPS predicted pi0 
pi0_p_HPS = []
pi0_eta_HPS = []
pi0_phi_HPS= []
rel_eta_HPS = []
rel_phi_HPS = []



# Generate predictions
i = 0
for elem in data_eval:
    x, y, PV, DM, HPSDM, HPS_pi0, jetpos = elem # import relevant info from DataLoader
    HPS_pi0 = np.array(HPS_pi0)[0]
    jetpos = np.array(jetpos[0])
    if DM ==1 or DM == 11: # Store the regressed pi0
        # Get truth and predictions
        y_pred = test(x, model)
        y = np.array(y)[0]
        # Store DMs
        genDM.append(np.array(DM)[0])
        HPSDM_list.append(np.array(HPSDM)[0])
        # Store RAW predictions
        rel_p.append(y[0])
        rel_eta.append(y[1])
        rel_phi.append(y[2])
        rel_p_pred.append(y_pred[0])
        rel_eta_pred.append(y_pred[1])
        rel_phi_pred.append(y_pred[2])
        # Find original eta and phi
        eta_orig = jetpos[0] + y[1]
        eta_orig_pred = jetpos[0] + y_pred[1] #CNN
        eta_orig_HPS = jetpos[0] + HPS_pi0[3]
        phi_orig = jetpos[1] + y[2]
        phi_orig_pred = jetpos[1] + y_pred[2]
        phi_orig_HPS = jetpos[1] + HPS_pi0[4]
        # Find original postions in xyz
        pos = pos_xyz(eta_orig, phi_orig)
        pos_pred = pos_xyz(eta_orig_pred, phi_orig_pred) # CNN 
        pos_HPS = pos_xyz(eta_orig_HPS, phi_orig_HPS)
        # Convert to pi0 (p, eta, phi)
        pi0 = pi0_xyz(pos, np.array(PV))[0]
        pi0_pred = pi0_xyz(pos_pred, np.array(PV))[0]
        pi0_HPS_ = pi0_xyz(pos_HPS, np.array(PV))[0]
        # Extract preditions
        pi0_eta.append(get_pi0_eta(pi0))
        pi0_phi.append(get_pi0_phi(pi0))
        pi0_eta_HPS.append(get_pi0_eta(pi0_HPS_))
        pi0_phi_HPS.append(get_pi0_phi(pi0_HPS_))
        pi0_eta_pred.append(get_pi0_eta(pi0_pred))
        pi0_phi_pred.append(get_pi0_phi(pi0_pred))
        pi0_p_HPS.append(get_HPS_p(HPS_pi0))
        rel_eta_HPS.append(HPS_pi0[3])
        rel_phi_HPS.append(HPS_pi0[4])
        i+=1
        if i%100 == 0:
            pbar.update(100)
    else: # only DMs with 1 pi0 treated
        i+=1
        if i%100 ==0:
            pbar.update(100)
        continue

print(f"Total of {len(rel_p)} DM 1 or 11 taus evaluated")


df = pd.DataFrame()
df["DM"] = genDM
df["HPSDM"] = HPSDM_list
df["relp"] = rel_p 
df["releta"] = rel_eta
df["relphi"] = rel_phi
df["relp_pred"] = rel_p_pred
df["releta_pred"] = rel_eta_pred
df["relphi_pred"] = rel_phi_pred
df["pi0_phi"] = pi0_phi
df["pi0_eta"] = pi0_eta
df["pi0_phi_pred"] = pi0_phi_pred
df["pi0_eta_pred"] = pi0_eta_pred
df["pi0_p_HPS"] = pi0_p_HPS
df["rel_eta_HPS"] = rel_eta_HPS
df["rel_phi_HPS"] = rel_phi_HPS
df["pi0_eta_HPS"] = pi0_eta_HPS
df["pi0_phi_HPS"] = pi0_phi_HPS


print(df)

print("Predictions computed")

save_folder = path_to_artifacts + "/predictions"
if not os.path.exists(save_folder):
    os.makedirs(save_folder)

savepath = save_folder + "/kinematic_pred.pkl"

df.to_pickle(savepath)

print("Predictions saved in artifacts/predictions")

