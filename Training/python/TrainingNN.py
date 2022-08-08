from DataLoader import DataLoader
import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.backend as K
from tensorflow.keras import regularizers
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Input, Dense, Conv2D, Dropout, Activation, BatchNormalization, Flatten, \
                                    Concatenate, PReLU, MaxPooling2D



def conv_layer(prev_layer, filters, kernel_size=3, n=1):
    conv = Conv2D(filters, kernel_size, name="conv_{}".format(n),
                  kernel_initializer='he_uniform')(prev_layer) # kernel_regularizer=None (no reg for now)
    return conv

def pool_layer(prev_layer, poolgridsize, n):
    pool = MaxPooling2D(pool_size = poolgridsize, name="maxpooling_{}".format(n))(prev_layer)
    return pool

def create_model():
    input_layer = Input(name="input_image", shape=(3, None, 33, 33))
    conv1 = conv_layer(Input, 31, n=1)
    conv2 = conv_layer(conv1, 29, n=2)
    conv3 = conv_layer(conv2, 27, n=3)


