import tensorflow as tf
from DataLoader import DataLoader
import os
import yaml

os.environ["CUDA_VISIBLE_DEVICES"]="-1" # don't need to use GPU


with open("../configs/training.yaml") as file:
    training_cfg = yaml.full_load(file)
    print("Training Config Loaded")

# source_dir = "/home/hep/lcr119/ImageData/" 
# sample = "GluGluHToTauTau_M125"

# n_tau = 100
# n_batches = 200

load = DataLoader(training_cfg)

#separte for testing
gen = load.get_generator()
# print(next(gen))


print("Generator created")
type = (tf.float32, tf.float32)
shape = ((33, 33, 3), None)
# shape = (((33, 33), (33, 33), (33, 33)), None)


ds = tf.data.Dataset.from_generator(gen, output_types = type, output_shapes=shape).prefetch(tf.data.AUTOTUNE).batch(50, drop_remainder=True)
print("TF dataset")
tf.data.experimental.save(ds, "/home/hep/lcr119/TFData/testformat", compression = "GZIP")
print("Conversion Complete")
