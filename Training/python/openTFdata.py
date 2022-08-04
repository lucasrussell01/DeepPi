import tensorflow as tf

file = "/home/hep/lcr119/TFData/testformat"

ds = tf.data.experimental.load(file, compression="GZIP")

for elem in ds:
    print(elem)
    break