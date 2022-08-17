import tensorflow as tf
from tensorflow.keras.callbacks import Callback

class TauLosses:

    @staticmethod
    @tf.function
    def DecayMode_loss(target, output):
        loss = tf.keras.losses.categorical_crossentropy(target, output)
        return loss

    def Kinematic_loss(target, output):
        kin_output = output[1] # shape = (p, eta, phi)
        kin_target = target[1]
        a = 0.01 # energy factor
        b = 1 # eta factor
        c = 1 # phi factor
        loss = a*(kin_output[0] - kin_target[0])**2 + b*(kin_output[1] - kin_target[1])**2 + c*(kin_output[2] - kin_target[2])**2
        return loss


class EpochCheckpoint(Callback):
    def __init__(self, file_name_prefix):
        self.file_name_prefix = file_name_prefix

    def on_epoch_end(self, epoch, logs=None):
        self.model.save('{}_e{}.tf'.format(self.file_name_prefix, epoch),
                        save_format="tf")
        print("Epoch {} is ended.".format(epoch))