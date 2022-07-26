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


def test(x, model):
    y_pred = model(x, training=False)
    return np.array(y_pred)[0][0]

def get_HPS_p(p_vect):
    p = np.sqrt(p_vect[0]**2 + p_vect[1]**2 + p_vect[2]**2)
    return p

# Load training cfg
with open(f'{path_to_artifacts}/input_cfg/training_cfg.yaml') as file:
    training_cfg = yaml.full_load(file)
    print("Training Config Loaded")
training_cfg["Setup"]["input_dir"] = '/vols/cms/lcr119/Images/v2Images/Evaluation'
training_cfg["Setup"]["n_batches"] = args.n_tau
training_cfg["Setup"]["n_batches_val"] = 0
training_cfg["Setup"]["val_split"] = 0

print(training_cfg)

# Load evaluation dataset
dataloader = DataLoader(training_cfg)
gen_eval = dataloader.get_generator_v2(primary_set = True, mom_evaluation = True)

if training_cfg["Setup"]["HPS_features"]:
    print("Warning: Model was trained with HPS vars")
    input_shape = (((33, 33, 5), 29), None, 3, None, None, 5,  2)
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

pi0_p = []
pi0_p_pred = []
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
    elif HPSDM == 1 or HPSDM == 11: # make sure HPS has reco tau as same DM
        y_pred = test(x, model)
        y = np.array(y)[0]
        pi0_p.append(y)
        pi0_p_pred.append(y_pred)
        pi0_p_HPS.append(get_HPS_p(HPS_pi0))
        i+=1
        if i%10 ==0:
            pbar.update(10)
    else:
        i+=1
        if i%10 ==0:
            pbar.update(10)
        continue

print(f"Total of {len(pi0_p)} DM 1 or 11 taus evaluated")


df = pd.DataFrame()
df["pi0_p"] = pi0_p 
df["pi0_p_pred"] = pi0_p_pred
df["pi0_p_HPS"] = pi0_p_HPS


print(df)

print("Predictions computed")

save_folder = path_to_artifacts + "/predictions"
if not os.path.exists(save_folder):
    os.makedirs(save_folder)
savepath = save_folder + "/kinematicv2_pred_ggH.pkl"

df.to_pickle(savepath)

print("Predictions saved in artifacts/predictions")
