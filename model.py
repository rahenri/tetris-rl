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
    def __init__(self, name, board_shape):
        super().__init__(name=name)

        self._layers = [
            Reshape(list(board_shape) + [1], input_shape=board_shape, name=f"{name}/1"),
            Conv2D(32, 3, padding="same", activation="relu", name=f"{name}/3"),
            Conv2D(32, 3, padding="same", activation="relu", name=f"{name}/4"),
            MaxPooling2D(name=f"{name}/5"),
            Dropout(0.5, name=f"{name}/6"),
            Conv2D(64, 3, padding="same", activation="relu", name=f"{name}/7"),
            Conv2D(64, 3, padding="same", activation="relu", name=f"{name}/8"),
            MaxPooling2D(name=f"{name}/9"),
            Dropout(0.5, name=f"{name}/10"),
            Conv2D(128, 3, padding="same", activation="relu", name=f"{name}/11"),
            Conv2D(128, 3, padding="same", activation="relu", name=f"{name}/12"),
            MaxPooling2D(name=f"{name}/13"),
            Dropout(0.5, name=f"{name}/14"),
            Flatten(name=f"{name}/15"),
            Dense(1024, activation="relu", name=f"{name}/16"),
            Dense(1024, activation="relu", name=f"{name}/17"),
            Dense(1, activation="linear", name=f"{name}/18"),
        ]

    def call(self, x, training=False):
        for layer in self._layers:
            x = layer(x, training=training)
        return x
