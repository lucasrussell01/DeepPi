# mlflow logging inspired by DeepTau
from DataLoader import DataLoader
from setup_gpu import setup_gpu
from losses import TauLosses, EpochCheckpoint
import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.backend as K
from tensorflow.keras import regularizers
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Input, Dense, Conv2D, Dropout, Activation, BatchNormalization, Flatten, \
                                    Concatenate, PReLU, MaxPooling2D
from tensorflow.keras.callbacks import Callback, ModelCheckpoint, CSVLogger, LearningRateScheduler
import mlflow
mlflow.tensorflow.autolog(log_models=False)
import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf
import json
import os
from glob import glob
import numpy as np


class DeepPiv1Model(keras.Model):

    def __init__(self, *args, regress_kinematic=False, use_HPS=True, kDM=1, kKin=1, **kwargs):
        super().__init__(*args, **kwargs)
        self.regress_kinematic = regress_kinematic # Toggle this to true if kinematic regression present
        self.use_HPS = use_HPS # Toggle true if HPS mass variables present
        self.DM_loss = TauLosses.DecayMode_loss
        self.DM_loss_tracker = keras.metrics.Mean(name="DM_loss")
        if self.regress_kinematic:       
            self.Kin_loss = TauLosses.Kinematic_loss
            self.MSE_p = TauLosses.MAE_momentum
            self.MSE_eta = TauLosses.MAE_eta
            self.MSE_phi = TauLosses.MAE_phi
            self.RMSE_p = TauLosses.RMSE_momentum
            self.RMSE_eta = TauLosses.RMSE_eta
            self.RMSE_phi = TauLosses.RMSE_phi
            self.Kin_loss_tracker = keras.metrics.Mean(name="Kin_loss")
            self.MSE_p_tracker = keras.metrics.Mean(name="MSE_momentum")
            self.MSE_eta_tracker = keras.metrics.Mean(name="MSE_eta")
            self.MSE_phi_tracker = keras.metrics.Mean(name="MSE_phi")
            self.RMSE_p_tracker = keras.metrics.Mean(name="RMSE_momentum")
            self.RMSE_eta_tracker = keras.metrics.Mean(name="RMSE_eta")
            self.RMSE_phi_tracker = keras.metrics.Mean(name="RMSE_phi")
            self.kDM = kDM # importance of L_DM
            self.kKin = kKin # importance of L_Kin
            print(f"Decay Mode Importance: {self.kDM}, Kinematic Importance: {self.kKin}")
        self.DM_accuracy = tf.keras.metrics.CategoricalAccuracy(name='DM_accuracy', dtype=None)
        

    def train_step(self, data):
        # Unpack the data
        if self.regress_kinematic:
            x, yDM, yKin, w = data
        else:
            x, yDM = data
        n_tau = tf.shape(yDM)[0]

        if self.regress_kinematic: # If kinematic regression active
            # Forward Pass:
            with tf.GradientTape() as DM_tape, tf.GradientTape() as Kin_tape:
                y_pred = self(x, training=True)
                y_predDM = y_pred[0]
                y_predKin = y_pred[1]
                DM_loss_vec = self.DM_loss(yDM, y_predDM)
                DM_loss = tf.reduce_sum(DM_loss_vec)/tf.cast(n_tau, dtype=tf.float32)
                Kin_loss_vec = self.Kin_loss(yKin, y_predKin)
                Kin_loss = tf.reduce_sum(tf.multiply(Kin_loss_vec, w))/tf.reduce_sum(w) # only for DM 1, 11

            # Group TPs for different blocks:
            DM_layers = [var for var in self.trainable_variables if ("DM" in var.name)] # final DM layers
            Kin_layers = [var for var in self.trainable_variables if ("Kin" in var.name)] # final Kinematic layers
            common_layers = [var for var in self.trainable_variables if ("Kin" not in var.name and "DM" not in var.name)] # layers common to both

            # Compute gradients
            grad_DM_glob = DM_tape.gradient(DM_loss, common_layers + DM_layers) # for whole network wrt DM
            grad_Kin_glob = Kin_tape.gradient(Kin_loss, common_layers + Kin_layers) # for whole network wrt Kin
            grad_DM_final = grad_DM_glob[len(common_layers):] # final DM layers
            grad_Kin_final = grad_Kin_glob[len(common_layers):] # final Kin layers
            grad_common = [self.kDM*grad_DM_glob[i] + self.kKin*grad_Kin_glob[i] for i in range(len(common_layers))]

            # Update trainable parameters
            self.optimizer.apply_gradients(zip(grad_common + grad_DM_final + grad_Kin_final, common_layers + DM_layers + Kin_layers))

            # Compute individual MSE:
            MSE_p = tf.reduce_sum(tf.multiply(self.MSE_p(yKin, y_predKin), w))/tf.reduce_sum(w)
            MSE_eta = tf.reduce_sum(tf.multiply(self.MSE_eta(yKin, y_predKin), w))/tf.reduce_sum(w)
            MSE_phi = tf.reduce_sum(tf.multiply(self.MSE_phi(yKin, y_predKin), w))/tf.reduce_sum(w)
            RMSE_p = tf.reduce_sum(tf.multiply(self.RMSE_p(yKin, y_predKin), w))/tf.reduce_sum(w)
            RMSE_eta = tf.reduce_sum(tf.multiply(self.RMSE_eta(yKin, y_predKin), w))/tf.reduce_sum(w)
            RMSE_phi = tf.reduce_sum(tf.multiply(self.RMSE_phi(yKin, y_predKin), w))/tf.reduce_sum(w)

        else: # If only DM classification
            # Forward pass:
            with tf.GradientTape() as DM_tape:
                y_predDM = self(x, training=True)
                DM_loss_vec = self.DM_loss(yDM, y_predDM)
                DM_loss = tf.reduce_sum(DM_loss_vec)/tf.cast(n_tau, dtype=tf.float32)
            # Compute gradients
            trainable_pars = self.trainable_variables
            gradients = DM_tape.gradient(DM_loss, trainable_pars)
            # Update trainable parameters
            self.optimizer.apply_gradients(zip(gradients, trainable_pars))
        

        # Update metrics
        self.DM_loss_tracker.update_state(DM_loss)
        self.DM_accuracy.update_state(yDM, y_predDM)
        if self.regress_kinematic:
            self.Kin_loss_tracker.update_state(Kin_loss)
            self.MSE_p_tracker.update_state(MSE_p)
            self.MSE_eta_tracker.update_state(MSE_eta)
            self.MSE_phi_tracker.update_state(MSE_phi)
            self.RMSE_p_tracker.update_state(RMSE_p)
            self.RMSE_eta_tracker.update_state(RMSE_eta)
            self.RMSE_phi_tracker.update_state(RMSE_phi)

        # Return a dict mapping metric names to current value (printout)
        metrics_out =  {m.name: m.result() for m in self.metrics}
        return metrics_out
    
    def test_step(self, data):
        # Unpack the data 
        if self.regress_kinematic:
            x, yDM, yKin, w = data
        else:
            x, yDM = data
        n_tau = tf.shape(yDM)[0]

        # Evaluate Model
        if self.regress_kinematic:
            y_pred = self(x, training=False)
            y_predDM = y_pred[0]
            y_predKin = y_pred[1]
            Kin_loss_vec = self.Kin_loss(yKin, y_predKin)
            Kin_loss = tf.reduce_sum(tf.multiply(Kin_loss_vec, w))/tf.reduce_sum(w) # only for DM 1, 11
            MSE_p = tf.reduce_sum(tf.multiply(self.MSE_p(yKin, y_predKin), w))/tf.reduce_sum(w)
            MSE_eta = tf.reduce_sum(tf.multiply(self.MSE_eta(yKin, y_predKin), w))/tf.reduce_sum(w)
            MSE_phi = tf.reduce_sum(tf.multiply(self.MSE_phi(yKin, y_predKin), w))/tf.reduce_sum(w)
            RMSE_p = tf.reduce_sum(tf.multiply(self.RMSE_p(yKin, y_predKin), w))/tf.reduce_sum(w)
            RMSE_eta = tf.reduce_sum(tf.multiply(self.RMSE_eta(yKin, y_predKin), w))/tf.reduce_sum(w)
            RMSE_phi = tf.reduce_sum(tf.multiply(self.RMSE_phi(yKin, y_predKin), w))/tf.reduce_sum(w)
        else:
            y_predDM = self(x, training=False)
        DM_loss_vec = self.DM_loss(yDM, y_predDM)
        DM_loss = tf.reduce_sum(DM_loss_vec)/tf.cast(n_tau, dtype=tf.float32)
        
        
        # Update the metrics 
        self.DM_loss_tracker.update_state(DM_loss)
        self.DM_accuracy.update_state(yDM, y_predDM)
        if self.regress_kinematic:
            self.Kin_loss_tracker.update_state(Kin_loss)
            self.MSE_p_tracker.update_state(MSE_p)
            self.MSE_eta_tracker.update_state(MSE_eta)
            self.MSE_phi_tracker.update_state(MSE_phi)
            self.RMSE_p_tracker.update_state(RMSE_p)
            self.RMSE_eta_tracker.update_state(RMSE_eta)
            self.RMSE_phi_tracker.update_state(RMSE_phi)

        # Return a dict mapping metric names to current value
        metrics_out = {m.name: m.result() for m in self.metrics}
        return metrics_out
    
    @property
    def metrics(self):
        # define metrics here so that `reset_states()` can be
        # called automatically at the start of each epoch
        # or at the start of `evaluate()`
        metrics = []
        metrics.append(self.DM_loss_tracker) 
        if self.regress_kinematic:
            metrics.append(self.Kin_loss_tracker)
        metrics.append(self.DM_accuracy) 
        if self.regress_kinematic:
            metrics.append(self.MSE_p_tracker)
            metrics.append(self.MSE_eta_tracker)
            metrics.append(self.MSE_phi_tracker)
            metrics.append(self.RMSE_p_tracker)
            metrics.append(self.RMSE_eta_tracker)
            metrics.append(self.RMSE_phi_tracker)

        for l in self._flatten_layers():
            metrics.extend(l._metrics)  # pylint: disable=protected-access
        return metrics


class DeepPiv2Model(keras.Model):

    def __init__(self, *args, use_weights = False, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_weights = use_weights
        self.Kin_loss = TauLosses.Kinematic_loss
        self.Kin_loss_tracker = keras.metrics.Mean(name="loss")
        self.MAE_p = TauLosses.MAE_momentum
        self.MAE_eta = TauLosses.MAE_eta
        self.MAE_phi = TauLosses.MAE_phi
        self.MAE_p_tracker = keras.metrics.Mean(name="MAE_momentum")
        self.MAE_eta_tracker = keras.metrics.Mean(name="MAE_eta")
        self.MAE_phi_tracker = keras.metrics.Mean(name="MAE_phi")
        if self.use_weights:
            self.Kin_raw_loss_tracker =  keras.metrics.Mean(name="raw_loss")

    def train_step(self, data):
        # Unpack the data
        if self.use_weights:
            x, y, w = data
        else:
            x, y = data

        # Forward pass:
        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)#[:,0]
            loss_vec = self.Kin_loss(y, y_pred)
            # since shape is 1 here:
            raw_loss = loss_vec #tf.reduce_sum(tf.reduce_sum(loss_vec))/tf.cast(tf.shape(y)[0], dtype=tf.float32)
            if self.use_weights:
                loss = tf.reduce_sum(tf.reduce_sum(tf.multiply(loss_vec, w)))/tf.reduce_sum(w)
            else:
                loss = raw_loss
    
        # Compute gradients
        trainable_pars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_pars)

        # Update trainable parameters
        self.optimizer.apply_gradients(zip(gradients, trainable_pars))
        
        # Compute loss components (unweighted):
        MAE_p = self.MAE_p(y, y_pred)
        MAE_eta = self.MAE_eta(y, y_pred)
        MAE_phi = self.MAE_phi(y, y_pred)

        # Update metrics
        self.Kin_loss_tracker.update_state(loss)
        self.MAE_p_tracker.update_state(MAE_p)
        self.MAE_eta_tracker.update_state(MAE_eta)
        self.MAE_phi_tracker.update_state(MAE_phi)
        if self.use_weights:
            self.Kin_raw_loss_tracker.update_state(raw_loss)


        # Return a dict mapping metric names to current value (printout)
        metrics_out =  {m.name: m.result() for m in self.metrics}
        return metrics_out
    
    def test_step(self, data):
        # Unpack the data 
        if self.use_weights:
            x, y, w = data
        else:
            x, y = data

        # Evaluate Model
        
        y_pred = self(x, training=True)#[:,0] flatten if only momentum
        loss_vec = self.Kin_loss(y, y_pred)
        raw_loss = loss_vec #tf.reduce_sum(tf.reduce_sum(loss_vec))/tf.cast(tf.shape(y)[0], dtype=tf.float32) # loss vec if not dividing/mult mom
        if self.use_weights:
            loss = tf.reduce_sum(tf.reduce_sum(tf.multiply(loss_vec, w)))/tf.reduce_sum(w)
        else:
            loss = raw_loss
        
        # Compute loss components (unweighted):
        MAE_p = self.MAE_p(y, y_pred)
        MAE_eta = self.MAE_eta(y, y_pred)
        MAE_phi = self.MAE_phi(y, y_pred)

        # Update the metrics 
        self.Kin_loss_tracker.update_state(loss)
        self.MAE_p_tracker.update_state(MAE_p)
        self.MAE_eta_tracker.update_state(MAE_eta)
        self.MAE_phi_tracker.update_state(MAE_phi)
        if self.use_weights:
            self.Kin_raw_loss_tracker.update_state(raw_loss)

        # Return a dict mapping metric names to current value
        metrics_out = {m.name: m.result() for m in self.metrics}
        return metrics_out
    
    @property
    def metrics(self):
        # define metrics here so that `reset_states()` can be
        # called automatically at the start of each epoch
        # or at the start of `evaluate()`
        metrics = []
        metrics.append(self.Kin_loss_tracker) 
        metrics.append(self.MAE_p_tracker)
        metrics.append(self.MAE_eta_tracker)
        metrics.append(self.MAE_phi_tracker)
        if self.use_weights:
            metrics.append(self.Kin_raw_loss_tracker)
        if self._is_compiled:
            # Track `LossesContainer` and `MetricsContainer` objects
            # so that attr names are not load-bearing.
            if self.compiled_loss is not None:
                metrics += self.compiled_loss.metrics
            if self.compiled_metrics is not None:
                metrics += self.compiled_metrics.metrics
        for l in self._flatten_layers():
            metrics.extend(l._metrics)  # pylint: disable=protected-access
        return metrics


def layer_ending(layer, n, dim2d = True, dropout=0): #, activation, dropout_rate # add these from cfg later
    norm_layer = BatchNormalization(name="norm_{}".format(n))(layer)
    if dim2d: # if conv or pooling
        activation_layer = PReLU(shared_axes=[1, 2], name='activation_{}'.format(n))(norm_layer) # share image dims
    else:
        activation_layer = PReLU(name='activation_{}'.format(n))(norm_layer)
    if dropout!=0:
        final = Dropout(dropout, name="dropout_{}".format(n))(activation_layer)
        return final
    else: 
        return activation_layer 
   

def conv_block(prev_layer, channels, kernel_size=3, n=1, dropout=0):
    conv = Conv2D(channels, kernel_size, name="conv_{}".format(n),
                  activation='relu', kernel_initializer='he_uniform')(prev_layer) # kernel_regularizer=None (no reg for now)
    out = layer_ending(conv, n, dropout=dropout)
    return out

def pool_block(prev_layer, n=1, dropout=0):
    poolsize = 3
    pool = MaxPooling2D(pool_size = poolsize, strides=poolsize, name="maxpooling_{}".format(n))(prev_layer)
    out = layer_ending(pool, n, dropout=dropout)
    return out

def dense_block(prev_layer, size, n=1, dropout=0):
    dense = Dense(size, name="dense_{}".format(n), kernel_initializer='he_uniform')(prev_layer)
    out = layer_ending(dense, n, dim2d=False, dropout=dropout)
    return out

def create_v1_model(dataloader):

    dropout_rate = dataloader.dropout_rate
    # Input Image
    input_layer = Input(name="input_image", shape=(33, 33, 5))

    # Feature extracting convolutional layers:
    conv0 = conv_block(input_layer, 24, dropout=dropout_rate, kernel_size=1, n=0) #31
    conv1 = conv_block(conv0, 15, dropout=dropout_rate, kernel_size=3, n=1) #31
    conv2 = conv_block(conv1, 9, dropout=dropout_rate, n=2) #29 
    conv3 = conv_block(conv2, 9, dropout=dropout_rate, n=3) #27
    conv4 = conv_block(conv3, 9, dropout=dropout_rate, n=4) #25
    conv5 = conv_block(conv4, 9, dropout=dropout_rate, n=5) #23
    conv6 = conv_block(conv5, 9, dropout=dropout_rate, n=6) #21
    conv7 = conv_block(conv6, 9, dropout=dropout_rate, n=7) #19
    conv8 = conv_block(conv7, 9, dropout=dropout_rate, n=8) #17
    conv9 = conv_block(conv8, 9, dropout=dropout_rate, n=9) #15
    conv10 = conv_block(conv9, 9, dropout=dropout_rate, n=10) #13
    conv11 = conv_block(conv10, 9, dropout=dropout_rate,n=11) #11
    conv12 = conv_block(conv11, 9, dropout=dropout_rate, n=12) #9
    conv13 = conv_block(conv12, 9, dropout=dropout_rate, n=13) #7
    conv14 = conv_block(conv13, 9, dropout=dropout_rate, n=14) #5

    # Flatten inputs
    flat = Flatten(name="flatten")(conv14) # 75
    flat_size = 225 

    if dataloader.use_HPS: # add HPS dense layers
        print("Warning: Using HPS mass variables")
        input_HPS = Input(name="input_mass_vars", shape=(31)) # 13
        dense_mass1 = dense_block(input_HPS, 50, dropout=dropout_rate, n="_mass_1")
        dense_mass2 = dense_block(dense_mass1, 50, dropout=dropout_rate, n="_mass_2")
        dense_mass3 = dense_block(dense_mass2, 50, dropout=dropout_rate, n="_mass_3")
        concat = Concatenate()([flat, dense_mass3])
        dense_DM1 = dense_block(concat, flat_size, dropout=dropout_rate, n="_DM_1")
    else: 
        dense_DM1 = dense_block(flat, flat_size, dropout=dropout_rate, n="_DM_1")

    # Dense layers for pi0 number extrapolation
    dense_DM2 = dense_block(dense_DM1, flat_size, dropout=dropout_rate, n="_DM_2")
    dense_DM3 = dense_block(dense_DM2, flat_size, dropout=dropout_rate, n="_DM_3")
    dense_DM4 = dense_block(dense_DM3, 100, dropout=dropout_rate, n="_DM_4")
    dense_DM5 = dense_block(dense_DM4, 3, n="_DM_5")
    # Pi0 output (softmax)
    outputDM = Activation("softmax", name="output_DM")(dense_DM5)

    if dataloader.regress_kinematic:
        # Dense layers for kinematic extrapolation
        dense_Kin1 = dense_block(flat, flat_size, dropout=dropout_rate, n="_Kin_1")
        dense_Kin2 = dense_block(dense_Kin1, flat_size, dropout=dropout_rate, n="_Kin_2")
        dense_Kin3 = dense_block(dense_Kin2, flat_size, dropout=dropout_rate, n="_Kin_3")
        dense_Kin4 = dense_block(dense_Kin3, 100, dropout=dropout_rate, n="_Kin_4")
        dense_Kin5 = dense_block(dense_Kin4, 3, n="_Kin_5")
        # Kinematic output
        outputKin= Dense(3, name="output_Kin")(dense_Kin5)

    # create model
    if dataloader.regress_kinematic:
        model = DeepPiv1Model(input_layer, [outputDM, outputKin], name=dataloader.model_name, regress_kinematic=True,
                             kDM=dataloader.kDM, kKin=dataloader.kKin)
    elif dataloader.use_HPS:
        model = DeepPiv1Model([input_layer, input_HPS], outputDM, name=dataloader.model_name, use_HPS=True)
    else:
        model = DeepPiv1Model(input_layer, outputDM, name=dataloader.model_name, regress_kinematic=False)

    return model


def create_v2_model(dataloader):

    dropout_rate = dataloader.dropout_rate
    # Input Image
    input_layer = Input(name="input_image", shape=(33, 33, 5))

    # Feature extracting convolutional layers:
    conv0 = conv_block(input_layer, 24, dropout=dropout_rate, kernel_size=1, n=0) #31
    conv1 = conv_block(conv0, 15, dropout=dropout_rate, kernel_size=3, n=1) #31
    conv2 = conv_block(conv1, 9, dropout=dropout_rate, n=2) #29 
    conv3 = conv_block(conv2, 9, dropout=dropout_rate, n=3) #27
    conv4 = conv_block(conv3, 9, dropout=dropout_rate, n=4) #25
    conv5 = conv_block(conv4, 9, dropout=dropout_rate, n=5) #23
    conv6 = conv_block(conv5, 9, dropout=dropout_rate, n=6) #21
    conv7 = conv_block(conv6, 9, dropout=dropout_rate, n=7) #19
    conv8 = conv_block(conv7, 9, dropout=dropout_rate, n=8) #17
    conv9 = conv_block(conv8, 9, dropout=dropout_rate, n=9) #15
    conv10 = conv_block(conv9, 9, dropout=dropout_rate, n=10) #13
    conv11 = conv_block(conv10, 9, dropout=dropout_rate,n=11) #11
    conv12 = conv_block(conv11, 9, dropout=dropout_rate, n=12) #9
    conv13 = conv_block(conv12, 9, dropout=dropout_rate, n=13) #7
    conv14 = conv_block(conv13, 9, dropout=dropout_rate, n=14) #5 # NB: the second input is the number of filters

    # Flatten inputs
    flat = Flatten(name="flatten")(conv14) # 75
    flat_size = 225 

    if dataloader.use_HPS: # add HPS dense layers
        print("Warning: Using HPS mass variables")
        input_HPS = Input(name="input_mass_vars", shape=(31))
        dense_mass1 = dense_block(input_HPS, 50, dropout=dropout_rate, n="_mass_1")
        dense_mass2 = dense_block(dense_mass1, 50, dropout=dropout_rate, n="_mass_2")
        dense_mass3 = dense_block(dense_mass2, 50, dropout=dropout_rate, n="_mass_3")
        concat = Concatenate()([flat, dense_mass3])
        dense_Kin1 = dense_block(concat, flat_size, dropout=dropout_rate, n="_Kin_1")
    else: 
        dense_Kin1 = dense_block(flat, flat_size, dropout=dropout_rate, n="_Kin_1")

    # Dense layers for kinematic extrapolation    
    dense_Kin2 = dense_block(dense_Kin1, flat_size, dropout=dropout_rate, n="_Kin_2")
    dense_Kin3 = dense_block(dense_Kin2, flat_size, dropout=dropout_rate, n="_Kin_3")
    dense_Kin4 = dense_block(dense_Kin3, 100, dropout=dropout_rate, n="_Kin_4")
    # Kinematic output
    outputKin= Dense(3, name="output_Kin")(dense_Kin4)

    # create model
    if dataloader.use_HPS:
        model = DeepPiv2Model([input_layer, input_HPS], outputKin, use_weights=dataloader.use_weights, name=dataloader.model_name)
    else:
        model = DeepPiv2Model(input_layer, outputKin, use_weights=dataloader.use_weights, name=dataloader.model_name)

    return model

def compile_v1_model(model, regress_kinematic=False):

    opt = tf.keras.optimizers.Nadam(learning_rate=1e-4)

    model.compile(loss=None, optimizer=opt, metrics=None)
    # mlflow log
    if regress_kinematic:
        print("Warning: Training with kinematic regression")
        metrics = {'DM_loss': '', 'Kin_loss': '', 'DM_accuracy': '', 'MSE_momentum': '', 'MSE_eta': '', 'MSE_phi': '',}
    else:
        print("Warning: Training without kinematic regression")
        metrics = {}
    mlflow.log_dict(metrics, 'input_cfg/metric_names.json')

def compile_v2_model(model, use_weights = False):

    opt = tf.keras.optimizers.Nadam(learning_rate=1e-4)

    model.compile(loss=None, optimizer=opt, metrics=None)
    # mlflow log
    if use_weights:
        metrics = {'loss': '', 'raw_loss': ''}
    else:
        metrics = {}
    mlflow.log_dict(metrics, 'input_cfg/metric_names.json')

def run_training(model, data_loader):

    print(f"Warning: Model name for training is: {data_loader.model_name}")
    if data_loader.model_name == "DeepPi_v1":
        # load generators
        gen_train = data_loader.get_generator_v1(primary_set = True)
        gen_val = data_loader.get_generator_v1(primary_set = False)
        # define input shapes
        if data_loader.regress_kinematic:
            input_shape = ((33, 33, 5), None, 3, None)
            input_types = (tf.float32, tf.float32, tf.float32, tf.float32)
        elif data_loader.use_HPS:
            input_shape = (((33, 33, 5), 31), None)
            input_types = ((tf.float32, tf.float32), tf.float32)
        else:
            input_shape = ((33, 33, 5), None)
            input_types = (tf.float32, tf.float32)
    elif data_loader.model_name == "DeepPi_v2":
        # load generators
        gen_train = data_loader.get_generator_v2(primary_set = True)
        gen_val = data_loader.get_generator_v2(primary_set = False)
        # define input shapes
        if data_loader.use_weights:
            print("Warning: Weights active")
            if data_loader.use_HPS:
                input_shape = (((33, 33, 5), 31), 3, None)
                input_types = ((tf.float32, tf.float32), tf.float32, tf.float32)
            else:
                input_shape = ((33, 33, 5), 3, None)
                input_types = (tf.float32, tf.float32, tf.float32)
        else:
            if data_loader.use_HPS:
                input_shape = (((33, 33, 5), 31), 3)
                input_types = ((tf.float32, tf.float32), tf.float32)
            else:
                input_shape = ((33, 33, 5), 3)
                input_types = (tf.float32, tf.float32)

    # datasets from generators
    data_train = tf.data.Dataset.from_generator(
        gen_train, output_types = input_types, output_shapes = input_shape
        ).prefetch(tf.data.AUTOTUNE).batch(data_loader.n_tau, drop_remainder=True)#.take(data_loader.n_batches)
    data_val = tf.data.Dataset.from_generator(
        gen_val, output_types = input_types, output_shapes = input_shape
        ).prefetch(tf.data.AUTOTUNE).batch(data_loader.n_tau, drop_remainder=True)#.take(data_loader.n_batches_val)
    
    # logs/callbacks
    model_name = data_loader.model_name
    log_name = f"{model_name}_step"
    csv_log_file = "metrics.log"
    if os.path.isfile(csv_log_file):
        close_file(csv_log_file)
        os.remove(csv_log_file)
    csv_log = CSVLogger(csv_log_file, append=True)
    epoch_checkpoint = EpochCheckpoint(log_name)
    callbacks = [epoch_checkpoint, csv_log]

    # Run training
    if data_loader.n_batches == -1:
        fit = model.fit(data_train, validation_data = data_val, epochs = data_loader.n_epochs, 
                        initial_epoch = data_loader.epoch, callbacks = callbacks)
    else:
        fit = model.fit(data_train, validation_data = data_val, epochs = data_loader.n_epochs, 
                        steps_per_epoch = data_loader.n_batches, validation_steps = data_loader.n_batches_val,
                        initial_epoch = data_loader.epoch, callbacks = callbacks)

    model_path = f"{log_name}_final.tf"
    model.save(model_path, save_format="tf")

    # mlflow logs
    for checkpoint_dir in glob(f'{log_name}*.tf'):
         mlflow.log_artifacts(checkpoint_dir, f"model_checkpoints/{checkpoint_dir}")
    mlflow.log_artifacts(model_path, "model")
    mlflow.log_artifact(csv_log_file)
    mlflow.log_param('model_name', model_name)

    return fit

@hydra.main(config_path='.', config_name='hydra_train')
def main(cfg: DictConfig) -> None:
    # set up mlflow experiment id
    mlflow.set_tracking_uri(f"file://{to_absolute_path(cfg.path_to_mlflow)}")
    experiment = mlflow.get_experiment_by_name(cfg.experiment_name)
    if experiment is not None:
        run_kwargs = {'experiment_id': experiment.experiment_id}
        if cfg["pretrained"] is not None: # initialise with pretrained run, otherwise create a new run
            run_kwargs['run_id'] = cfg["pretrained"]["run_id"]
    else: # create new experiment
        experiment_id = mlflow.create_experiment(cfg.experiment_name)
        run_kwargs = {'experiment_id': experiment_id}
        

    # run the training with mlflow tracking
    with mlflow.start_run(**run_kwargs) as main_run:

        if cfg["pretrained"] is not None:
            mlflow.start_run(experiment_id=run_kwargs['experiment_id'], nested=True)
        active_run = mlflow.active_run()
        run_id = active_run.info.run_id
        
        # load configs
        setup_gpu(cfg.gpu_cfg)
        training_cfg = OmegaConf.to_object(cfg.training_cfg) # convert to python dictionary
        dataloader = DataLoader(training_cfg)

        # main training
        if dataloader.model_name == "DeepPi_v1":
            model = create_v1_model(dataloader)
        elif dataloader.model_name == "DeepPi_v2":
            model = create_v2_model(dataloader)

        if cfg.pretrained is None:
            print("Warning: no pretrained NN -> training will be started from scratch")
        else:
            print("Warning: training will be started from pretrained model.")
            print(f"Model: run_id={cfg.pretrained.run_id}, experiment_id={cfg.pretrained.experiment_id}, model={cfg.pretrained.starting_model}")

            path_to_pretrain = to_absolute_path(f'{cfg.path_to_mlflow}/{cfg.pretrained.experiment_id}/{cfg.pretrained.run_id}/artifacts/')
            old_model = load_model(path_to_pretrain+f"/model_checkpoints/{cfg.pretrained.starting_model}",
                compile=False, custom_objects = None)
            for layer in model.layers:
                weights_found = False
                for old_layer in old_model.layers:
                    if layer.name == old_layer.name:
                        layer.set_weights(old_layer.get_weights())
                        weights_found = True
                        break
                if not weights_found:
                    print(f"Weights for layer '{layer.name}' not found.")

        if dataloader.model_name == "DeepPi_v1":
            compile_v1_model(model, regress_kinematic=dataloader.regress_kinematic)
        elif dataloader.model_name == "DeepPi_v2":
            compile_v2_model(model, use_weights=dataloader.use_weights)

        fit = run_training(model, dataloader)

        # log NN params
        with open(to_absolute_path(f'{cfg.path_to_mlflow}/{run_kwargs["experiment_id"]}/{run_id}/artifacts/model_summary.txt')) as f:
            for l in f:
                if (s:='Trainable params: ') in l:
                    mlflow.log_param('n_train_params', int(l.split(s)[-1].replace(',', '')))

        # log training related files
        mlflow.log_dict(training_cfg, 'input_cfg/training_cfg.yaml')
        mlflow.log_artifact(to_absolute_path("TrainingNN.py"), 'input_cfg')

        # log hydra files
        mlflow.log_artifacts('.hydra', 'input_cfg/hydra')
        mlflow.log_artifact('TrainingNN.log', 'input_cfg/hydra')

        # log misc. info
        mlflow.log_param('run_id', run_id)
        print(f'\nTraining has finished! Corresponding MLflow experiment name (ID): {cfg.experiment_name}({run_kwargs["experiment_id"]}), and run ID: {run_id}\n')
        mlflow.end_run()


if __name__ == '__main__':
    main()