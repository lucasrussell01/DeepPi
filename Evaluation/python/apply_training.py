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
parser = argparse.ArgumentParser(description='Generate predictions for DeepPi training')
parser.add_argument('--expID', required=True, type=str, help="Experiment ID")
parser.add_argument('--runID', required=True, type=str, help="Run ID")
parser.add_argument('--n_tau', required=False, default = 50000, type=int, help="n_tau")

args = parser.parse_args()


path_to_mlflow = "../../Training/python/mlruns/"
expID = args.expID 
runID = args.runID

path_to_artifacts = path_to_mlflow + expID + "/" + runID + "/artifacts"

def test(data, model):
        # Unpack the data
        x, y = data
        y_pred = model(x, training=False) 
        return (y, y_pred)

# Load training cfg
with open(f'{path_to_artifacts}/input_cfg/training_cfg.yaml') as file:
    training_cfg = yaml.full_load(file)
    print("Training Config Loaded")
training_cfg["Setup"]["input_dir"] = '/vols/cms/lcr119/Images/09082022/Evaluation'
training_cfg["Setup"]["n_batches"] = args.n_tau # 250k is full because batch size 1
training_cfg["Setup"]["n_batches_val"] = 0
training_cfg["Setup"]["val_split"] = 0

print(training_cfg)

# Load evaluation dataset
dataloader = DataLoader(training_cfg)
gen_eval = dataloader.get_generator(primary_set = True, evaluation=True)
input_shape = ((33, 33, 3), None)
input_types = (tf.float32, tf.float32)
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

full_pred = []
truth = []
truthDM = []
max_pred = []

pbar = tqdm(total = dataloader.n_batches)

i = 0
for elem in data_eval:
    y, y_pred = test(elem, model)

    # below is code for if dataloader not in eval mode
    # onehot_truth.append(y)
    full_pred.append(y_pred)
    # if out[0][0][0] == 1:
    #     truth.append(0)
    # elif out[0][0][1] == 1:
    #     truth.append(1)
    # elif out[0][0][2] == 1:
    #     truth.append(2)
    if y == 0 or y ==10:
        truth.append(0)
    elif y == 1 or y == 11:
        truth.append(1)
    elif y == 2:
        truth.append(2)
    else:
        raise RuntimeError("Unknown DM")
    truthDM.append(int(y))
    max_pred.append(int(np.where(y_pred[0] == np.max(y_pred[0]))[0]))
    i+=1
    if i%10 ==0:
        pbar.update(10)

df = pd.DataFrame()
df["truth"] = truth
df["max_pred"] = max_pred
df["truthDM"] = truthDM
df["full_pred"] = full_pred

print(df)

print("Predictions computed")

save_folder = path_to_artifacts + "/predictions"
if not os.path.exists(save_folder):
    os.makedirs(save_folder)
savepath = save_folder + "/pred_ggH.pkl"
df.to_pickle(savepath)

print("Predictions saved in artifacts/predictions")