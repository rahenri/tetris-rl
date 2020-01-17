import tensorflow as tf
import numpy as np

from tensorflow.keras.layers import (
    Dense,
    Flatten,
    Conv2D,
    MaxPooling2D,
    Dropout,
    Reshape,
    Activation,
)
from tensorflow import keras


class Block(keras.Model):
    def __init__(self, name, filters, layers, max_pool=False):
        super().__init__(name=name)

        self._layers = []

        for i in range(layers):
            self._layers.append(
                Conv2D(
                    filters,
                    3,
                    padding="same",
                    activation="relu",
                    name=f"{name}/conv_{i}",
                )
            )
        if max_pool:
            self._layers.append(MaxPooling2D(name=f"{name}/3"))

    def call(self, tensor, training=False):
        for layer in self._layers:
            tensor = layer(tensor, training=training)
        return tensor


class Model(keras.Model):
    def __init__(self, name, board_shape, output_shape):
        super().__init__(name=name)

        self.blocks = []

        # self.blocks = [
        #     Reshape(
        #         list(board_shape) + [1], input_shape=board_shape, name=f"{name}/reshape"
        #     )
        # ]

        # for i, size in enumerate([32, 64]):
        #     self.blocks.append(Block(name=f"{name}/block_{i}", filters=size, layers=2))

        self.blocks.append(Flatten(name=f"{name}/flatten"))

        for i, units in enumerate([512, 512, 512]):
            self.blocks.append(
                Dense(units, activation="relu", name=f"{name}/dense_{i+1}")
            )

        self.blocks.append(
            Dense(output_shape, activation="sigmoid", name=f"{name}/readout")
        )

        inputs = tf.convert_to_tensor(np.zeros([1] + list(board_shape), dtype=np.int32))
        self(inputs)

    @tf.function
    def call(self, tensor, training=False):
        tensor = tf.one_hot(tensor, 3)
        for layer in self.blocks:
            tensor = layer(tensor, training=training)
        return tensor * 2 - 1
