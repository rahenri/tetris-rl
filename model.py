from tensorflow.keras.layers import (
    Dense,
    Flatten,
    Conv2D,
    MaxPooling2D,
    Dropout,
    Reshape,
)
from tensorflow.keras import Model


class AgentModel(Model):
    def __init__(self, board_shape):
        super().__init__()

        self._layers = [
            Reshape(list(board_shape) + [1], input_shape=board_shape),
            Conv2D(32, 3, padding="same", activation="relu"),
            Conv2D(32, 3, padding="same", activation="relu"),
            MaxPooling2D(),
            Dropout(0.5),
            Conv2D(64, 3, padding="same", activation="relu"),
            Conv2D(64, 3, padding="same", activation="relu"),
            MaxPooling2D(),
            Dropout(0.5),
            Conv2D(128, 3, padding="same", activation="relu"),
            Conv2D(128, 3, padding="same", activation="relu"),
            MaxPooling2D(),
            Dropout(0.5),
            Flatten(),
            Dense(1024, activation="relu"),
            Dense(1024, activation="relu"),
            Dense(1, activation="linear"),
        ]

    def call(self, x, training=False):
        for layer in self._layers:
            x = layer(x, training=training)
        return x
