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
        a = 0.01 # energy factor
        b = 1 # eta factor
        c = 1 # phi factor
        loss = a*(output[:, 0] - target[:, 0])**2 + b*(output[:, 1] - target[:, 1])**2 + c*(output[:, 2] - target[:, 2])**2
        return loss


class EpochCheckpoint(Callback):
    def __init__(self, file_name_prefix):
        self.file_name_prefix = file_name_prefix

    def on_epoch_end(self, epoch, logs=None):
        self.model.save('{}_e{}.tf'.format(self.file_name_prefix, epoch),
                        save_format="tf")
        print("Epoch {} is ended.".format(epoch))