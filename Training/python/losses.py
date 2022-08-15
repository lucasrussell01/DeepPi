import tensorflow as tf
from tensorflow.keras.callbacks import Callback

class TauLosses:

    @staticmethod
    @tf.function
    def DecayMode_loss(target, output):
        loss = tf.keras.losses.categorical_crossentropy(target, output)
        return loss

class EpochCheckpoint(Callback):
    def __init__(self, file_name_prefix):
        self.file_name_prefix = file_name_prefix

    def on_epoch_end(self, epoch, logs=None):
        self.model.save('{}_e{}.tf'.format(self.file_name_prefix, epoch),
                        save_format="tf")
        print("Epoch {} is ended.".format(epoch))