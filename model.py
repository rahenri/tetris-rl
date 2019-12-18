from tensorflow.keras.layers import (
    Dense,
    Flatten,
    Conv2D,
    MaxPooling2D,
    Dropout,
    Reshape,
)
from tensorflow.keras import Model


class Block(Model):
    def __init__(self, name, filters, layers):
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
        self._layers.append(MaxPooling2D(name=f"{name}/3"))

    def call(self, x, training=False):
        for layer in self._layers:
            x = layer(x, training=training)
        return x


class AgentModel(Model):
    def __init__(self, name, board_shape):
        super().__init__(name=name)

        self._layers = [
            Reshape(
                list(board_shape) + [1], input_shape=board_shape, name=f"{name}/reshape"
            )
        ]

        for i, size in enumerate([32, 64, 128]):
            self._layers.append(Block(name=f"{name}/block_{i}", filters=size, layers=4))

        self._layers.extend(
            [
                Flatten(name=f"{name}/15"),
                Dense(1024, activation="relu", name=f"{name}/dense_1"),
                Dense(1024, activation="relu", name=f"{name}/dense_2"),
                Dense(1, activation="linear", name=f"{name}/readout"),
            ]
        )

    def call(self, x, training=False):
        for layer in self._layers:
            x = layer(x, training=training)
        return x
