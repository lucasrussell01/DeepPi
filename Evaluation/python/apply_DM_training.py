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
parser = argparse.ArgumentParser(description='Generate DM predictions for DeepPi training')
parser.add_argument('--expID', required=True, type=str, help="Experiment ID")
parser.add_argument('--runID', required=True, type=str, help="Run ID")
parser.add_argument('--n_tau', required=False, default = 50000, type=int, help="n_tau")
parser.add_argument('--HPS', required=False, default = "True", type=str, help="use HPS taus only")

args = parser.parse_args()

if args.HPS=="True":
    print("WARNING: Selecting HPS taus only")

path_to_mlflow = "../../Training/python/mlruns/"
expID = args.expID 
runID = args.runID

path_to_artifacts = path_to_mlflow + expID + "/" + runID + "/artifacts"

def test(x, model):
    y_pred = model(x, training=False) 
    return (y_pred)

def DMtopi0(DM): # convert DM to 0, 1 or 2 pi0s
    if DM == 0 or DM == 10:
        return 0
    elif DM == 1 or DM ==11:
        return 1
    elif DM == 2:
        return 2
    elif DM == -1:
        return -1

# Load training cfg
with open(f'{path_to_artifacts}/input_cfg/training_cfg.yaml') as file:
    training_cfg = yaml.full_load(file)
    print("Training Config Loaded")
training_cfg["Setup"]["input_dir"] = '/vols/cms/lcr119/Images/Images_MVA/Evaluation'
training_cfg["Setup"]["n_batches"] = args.n_tau # 250k is full because batch size 1
training_cfg["Setup"]["n_batches_val"] = 0
training_cfg["Setup"]["val_split"] = 0

print(training_cfg)

# Load evaluation dataset
dataloader = DataLoader(training_cfg)
gen_eval = dataloader.get_generator_v1(primary_set = True, DM_evaluation=True)

if training_cfg["Setup"]["HPS_features"]:
    input_shape = (((33, 33, 5), 31), None, None, None, None, None, None)
    input_types = ((tf.float32, tf.float32), tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32)
else:
    input_shape = ((33, 33, 5), None, None, None)
    input_types = (tf.float32, tf.float32, tf.float32, tf.float32)
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

# full_pred = []
truth = []
truthDM = []
CNN_pred = []
HPS_pred = []
MVA_pred = []
VSjet = []
VSmu = []
VSe = []

pbar = tqdm(total = dataloader.n_batches)

i = 0
for elem in data_eval:
    x, y, yHPSDM, yMVADM, vsj, vse, vsmu = elem
    if args.HPS=="True"and yHPSDM not in [0, 1, 2, 10, 11] or yMVADM==-1:
        continue
    else:
        y_pred = test(x, model)
        if training_cfg["Setup"]["kinematic"]:
            y_pred = y_pred[0] # take only DM output

        # full_pred.append(y_pred)
        if y == 0 or y ==10:
            truth.append(0)
        elif y == 1 or y == 11:
            truth.append(1)
        elif y == 2:
            truth.append(2)
        else:
            raise RuntimeError("Unknown DM")
        truthDM.append(int(y))
        CNN_pred.append(int(np.where(y_pred[0] == np.max(y_pred[0]))[0]))
        HPS_pred.append(DMtopi0(np.array(yHPSDM)[0]))
        MVA_pred.append(DMtopi0(np.array(yMVADM)[0]))
        VSjet.append(np.array(vsj)[0])
        VSe.append(np.array(vse)[0])
        VSmu.append(np.array(vsmu)[0])

    i+=1
    if i%10 ==0:
        pbar.update(10)

df = pd.DataFrame()
df["truthDM"] = truthDM
df["truth"] = truth
df["CNN_pred"] = CNN_pred
df["MVA_pred"] = MVA_pred
df["HPS_pred"] = HPS_pred
df["VSjet"] = VSjet
df["VSe"] = VSe
df["VSmu"] = VSmu
# df["full_pred"] = full_pred

print(df)

print("Predictions computed")

save_folder = path_to_artifacts + "/predictions"
if not os.path.exists(save_folder):
    os.makedirs(save_folder)
if args.HPS=="True":
    savepath = save_folder + "/pred_DM_HPS_only.pkl"
else:
    savepath = save_folder + "/pred_DM.pkl"

df.to_pickle(savepath)

print("Predictions saved in artifacts/predictions")