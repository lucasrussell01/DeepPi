import tensorflow as tf
from tensorflow.keras.models import load_model
import json
import yaml
import os
import sys
sys.path.append("../../Training/python")
from DataLoader import DataLoader
os.environ["CUDA_VISIBLE_DEVICES"]="-1" # don't need to use GPU

path_to_mlflow = "../../Training/python/mlruns/"
expID = "0"
runID = "1cd5023113ea4637969ca51fbf531cc1"

path_to_artifacts = path_to_mlflow + expID + "/" + runID + "/artifacts"


# Load training cfg
with open(f'{path_to_artifacts}/input_cfg/training_cfg.yaml') as file:
    training_cfg = yaml.full_load(file)
    print("Training Config Loaded")
training_cfg["input_dir"] = '/vols/cms/lcr119/Images/09082022/Evaluation'
training_cfg["n_batches"] = 5000
training_cfg["n_batches_val"] = 0
training_cfg["val_split"] = 0

# Load evaluation dataset
dataloader = DataLoader(training_cfg)
gen_eval = dataloader.get_generator(primary_set = True)
input_shape = ((33, 33, 3), None)
input_types = (tf.float32, tf.float32)
data_eval = tf.data.Dataset.from_generator(
    gen_eval, output_types = input_types, output_shapes = input_shape
    ).prefetch(tf.data.AUTOTUNE).batch(dataloader.n_tau).take(dataloader.n_batches)
print("Dataset Loaded")

# Load model
with open(f'{path_to_artifacts}/input_cfg/metric_names.json') as f:
    metric_names = json.load(f)
    path_to_model = f'{path_to_artifacts}/model'

model = load_model(path_to_model, {name: lambda _: None for name in metric_names.keys()})
print("Model loaded")


def test(data, model):
        # Unpack the data
        x, y= data
        y = y.numpy()[:,0] # Data/MC truth stored at this index (cf dataloader)
        if args.not_adversarial:
            # Only classification predictions are available
            y_pred_class = model(x, training=False) 
            return y_pred_class, y, sample_weight
        else:
            # Compute predictions for classification and adversarial subnetworks
            y_pred = model(x, training=False)
            y_pred_class = y_pred[0].numpy() # classification prediction
            y_pred_adv = y_pred[1].numpy() # y_adv prediction
            return y_pred_class, y_pred_adv, y, sample_weight