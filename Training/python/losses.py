import tensorflow as tf
from tensorflow.keras.callbacks import Callback

class TauLosses:

    @staticmethod
    @tf.function
    def DecayMode_loss(target, output):
        loss = tf.keras.losses.categorical_crossentropy(target, output)
        return loss

    @staticmethod
    @tf.function
    def Kinematic_loss(target, output):
        # loss = TauLosses.MAE_momentum(target, output) 
        # pos_pred = tf.sqrt(tf.square(output[:, 1]) + tf.square(output[:, 2]))
        # pos_true = tf.sqrt(tf.square(target[:, 1]) + tf.square(target[:, 2]))
        # loss = tf.keras.losses.mean_absolute_error(pos_true, pos_pred)
        # loss = tf.sqrt(tf.square(output[:, 1]-target[:, 1]) + tf.square(output[:, 2]-target[:, 2]))/tf.constant([50.0])
        loss = TauLosses.MAE_eta(target, output) + TauLosses.MAE_phi(target, output)# + TauLosses.MAE_momentum(target, output) 
        return loss

    @staticmethod
    @tf.function
    def MAE_momentum(target, output):
        loss = tf.keras.losses.mean_absolute_error(target[:, 0], output[:, 0])
        return loss
    
    @staticmethod
    @tf.function
    def MAE_eta(target, output):
        loss = tf.keras.losses.mean_absolute_error(target[:, 1], output[:, 1])
        return loss

    @staticmethod
    @tf.function
    def MAE_phi(target, output):
        loss = tf.keras.losses.mean_absolute_error(target[:, 2], output[:, 2])
        return loss

    @staticmethod
    @tf.function
    def MSE_eta(target, output):
        loss = tf.keras.losses.mean_squared_error(target[:, 1], output[:, 1])#*tf.math.maximum(tf.constant([2.0]), target[:, 0]) # shape: 1
        return loss

    @staticmethod
    @tf.function
    def MSE_phi(target, output):
        loss = tf.keras.losses.mean_squared_error(target[:, 2], output[:, 2])#*tf.math.maximum(tf.constant([2.0]), target[:, 0]) # shape: 1
        return loss

    @staticmethod
    @tf.function
    def RMSE_momentum(target, output):
        loss = ((output[:, 0] - target[:, 0])**2)**0.5
        return loss
    
    @staticmethod
    @tf.function
    def RMSE_eta(target, output):
        loss = ((output[:, 1] - target[:, 1])**2)**0.5 
        return loss

    @staticmethod
    @tf.function
    def RMSE_phi(target, output):
        loss = ((output[:, 2] - target[:, 2])**2)**0.5 
        return loss

    @staticmethod
    @tf.function
    def MSE_momentum_v2(target, output):
        loss = tf.keras.losses.mean_absolute_error(target, output)/tf.math.maximum(tf.constant([2.0]), target)
        return loss


class EpochCheckpoint(Callback):
    def __init__(self, file_name_prefix):
        self.file_name_prefix = file_name_prefix

    def on_epoch_end(self, epoch, logs=None):
        self.model.save('{}_e{}.tf'.format(self.file_name_prefix, epoch),
                        save_format="tf")
        print("Epoch {} is ended.".format(epoch))