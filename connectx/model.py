import tensorflow as tf
import numpy as np
from observation import ObservationVector, SingleObservation

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
    def __init__(self, name, board_shape):
        super().__init__(name=name)

        self.blocks = []

        # self.blocks = [
        #     Reshape(
        #         list(board_shape) + [1], input_shape=board_shape, name=f"{name}/reshape"
        #     )
        # ]

        # for i, size in enumerate([32, 64]):
        #     self.blocks.append(Block(name=f"{name}/block_{i}", filters=size, layers=2))

        self.blocks.append(Flatten(name=f"{name}/board/flatten"))

        for i, units in enumerate([512, 512, 512]):
            self.blocks.append(
                Dense(units, activation="relu", name=f"{name}/board/dense_{i+1}")
            )

        self.turn_model = Dense(512, activation="linear", name=f"{name}/turn")

        self.tail = []
        for i, units in enumerate([512, 512, 512]):
            self.tail.append(Dense(units, activation="relu", name=f"{name}/tail_{i+1}"))

        self.readout = Dense(1, activation="sigmoid", name=f"{name}/readout")

        board = tf.convert_to_tensor(np.zeros(board_shape, dtype=np.int32))
        inputs = ObservationVector.from_list(board_shape, [SingleObservation(board, 0)])
        self(inputs)

    @tf.function
    def call_fn(self, boards, turns, training=False):
        tensor = tf.one_hot(boards, 3)
        for layer in self.blocks:
            tensor = layer(tensor, training=training)

        turns = tf.cast(tf.reshape(turns, [-1, 1]), tf.float32)
        turns = self.turn_model(turns, training=training)

        tensor = turns * tensor

        for layer in self.tail:
            tensor = layer(tensor, training=training)

        tensor = self.readout(tensor, training=training)

        return tensor * 2 - 1

    def call(self, obs, training=False):
        return self.call_fn(
            tf.convert_to_tensor(obs.boards, tf.int32),
            tf.convert_to_tensor(obs.turns, tf.float32),
            training,
        )
