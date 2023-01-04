import keras.backend as K
from keras.layers import Layer

class ElliotActivation(Layer):
    def call(self, inputs):
        return inputs / (1 + K.abs(inputs))

    def get_config(self):
        base_config = super(ElliotActivation, self).get_config()
        return base_config
