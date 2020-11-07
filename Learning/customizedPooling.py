from tensorflow.keras.layers import Layer
import numpy as np
import tensorflow as tf


class CustomizedPooling(Layer):
    def __init__(self, reduce_function=1,  **kwargs):
        self.reduce_function = reduce_function
        # Bsp. gewicht hinzufügen
        # self.w = self.add_weight(
        #    shape=(input_dim, units), initializer="random_normal", trainable=True
        # )
        super(CustomizedPooling, self).__init__(**kwargs)

    # vielleicht gewichte statt pooling mean bzw. max erstmal aber nicht
    # def build(self, input_shape):
    #    self.kernel = self.add_weight(name='kernel',
    #                                  shape=(input_shape[1], self.output_dim),
    #                                  initializer='normal', trainable=True)
    #    super(CustomizedPooling, self).build(input_shape)

    def call(self, input_data, **kwargs):
        shape = input_data.shape
        neuer_tensor = np.empty((1, 3, 3, 0), dtype=np.float32)
        erstes = True
        position = 0
        while position < shape[3]:
            if position < (shape[3] - 3):
                kernel = input_data[:, :, :, position:(position + 3)]
            else:
                kernel = input_data[:, :, :, position:(shape[3] + 1)]
            position += 3
            if self.reduce_function:
                spalte_neu = tf.compat.v1.reduce_mean(kernel, axis=[3], keepdims=True)
            else:
                spalte_neu = tf.compat.v1.reduce_max(kernel, axis=[3], keepdims=True)
            if erstes:
                neuer_tensor = spalte_neu
                erstes = False
            else:
                neuer_tensor = tf.concat([neuer_tensor, spalte_neu], axis=3)
        return neuer_tensor

    def get_config(self):
        config = super().get_config().copy()
        return config
