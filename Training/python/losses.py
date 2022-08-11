import tensorflow as tf

class TauLosses:

    @staticmethod
    @tf.function
    def xentropyloss(target, output):
        loss = tf.keras.losses.categorical_crossentropy(target, output)
        return loss

