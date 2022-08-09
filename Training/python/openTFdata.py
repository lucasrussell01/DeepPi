import tensorflow as tf
import os

os.environ["CUDA_VISIBLE_DEVICES"]="-1" # don't need to use GPU

file = "/home/hep/lcr119/TFData/testformat"

ds = tf.data.experimental.load(file, compression="GZIP")

for elem in ds:
    for i in elem:
        print(tf.shape(i))
        # print(i)
    break