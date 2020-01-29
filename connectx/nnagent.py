import random

import tensorflow as tf
import numpy as np

from model import Model
from observation import SingleObservation, ObservationVector


def prod(values):
    out = 1
    for v in values:
        out *= v
    return out


class NNAgent:
    def __init__(self, board_shape):
        self.learning_rate_var = tf.Variable(0.001)

        self.board_shape = board_shape
        self.gamma = 0.95
        self.episilon = 0.1
        self.lamb = 0.98

        self.board_shape = board_shape

        self.value_net = Model("value", board_shape)
        self.target_net = Model("target", board_shape)

        weights = 0
        for var, var_target in zip(
            self.value_net.trainable_variables, self.target_net.trainable_variables,
        ):
            var.assign(var_target)
            w = prod(var.shape)
            weights += w

        for var in self.target_net.trainable_variables:
            w = prod(var.shape)
            print(f"{var.name}: {w} ({w / weights * 100:.1f}%)")
        print(f"Weights: {weights}")

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate_var)
        self.loss_function = tf.keras.losses.MeanSquaredError()

    def act(self, env, randomize=False, verbose=False):
        actions = env.board.list_moves()
        assert len(actions) > 0

        if randomize and random.random() < self.episilon:
            return random.choice(actions), 0

        obs = []
        rewards = []
        endeds = []
        player = env.turn()
        for action in actions:
            env_copy = env.copy()
            reward = 0
            if env_copy.would_win(action, player):
                reward += 1000000
            elif env_copy.would_win(action, 3 - player):
                reward += 1000
            _, ended = env_copy.step(action)
            obs.append(env_copy.obs())
            rewards.append(reward)
            endeds.append(ended)

        endeds = np.array(endeds, dtype=np.float32)
        rewards = np.array(rewards, dtype=np.float32)
        obs = ObservationVector.from_list(self.board_shape, obs)
        scores = rewards - self.target_net(obs).numpy().reshape(-1) * (1.0 - endeds)

        if verbose:
            print(list(zip(actions, scores)))

        best_score = None
        best_action = None
        for action, score in zip(actions, scores):
            if best_score is None or score > best_score:
                best_score = score
                best_action = action

        return best_action, best_score

    def _make_features(self, memory, batch_size):
        obs, next_obs, rewards, dones = memory.sample(batch_size)

        not_dones = tf.convert_to_tensor(1 - dones, dtype=tf.float32)
        rewards = tf.convert_to_tensor(rewards, dtype=tf.float32)

        predictions = tf.reshape(self.target_net(next_obs), [-1])
        targets = rewards - predictions * self.gamma * not_dones

        return obs, targets

    def trainable_variables(self):
        return self.value_net.trainable_variables + self.target_net.trainable_variables

    def save_model(self, filename):
        tensors = {v.name: v.numpy() for v in self.trainable_variables()}
        np.savez(filename, **tensors)

    def load_model(self, filename):
        tensors = np.load(filename)
        for v in self.trainable_variables():
            name = v.name
            v.assign(tensors[name])

    def train(self, memory, batch_size):
        obs, target = self._make_features(memory, batch_size)

        with tf.GradientTape() as tape:
            predicions = self.value_net(obs, training=True)
            loss = self.loss_function(target, predicions)
        gradients = tape.gradient(loss, self.value_net.trainable_variables)
        self.optimizer.apply_gradients(
            zip(gradients, self.value_net.trainable_variables)
        )
        for a, b in zip(
            self.value_net.trainable_variables, self.target_net.trainable_variables,
        ):
            b.assign(b * self.lamb + a * (1.0 - self.lamb))
        del tape

        sample_obs = ObservationVector.from_list(
            self.board_shape,
            [SingleObservation(np.zeros(self.board_shape, dtype=np.int8), 0)],
        )

        total_weigth = 0
        for variable in self.value_net.trainable_variables:
            total_weigth += tf.reduce_sum(variable * variable)
        total_weigth = np.sqrt(total_weigth.numpy())

        sample_score = self.target_net(sample_obs).numpy().reshape(-1)[0]

        return {
            "loss": loss.numpy(),
            "mean target value": target.numpy().mean(),
            "sample score": sample_score,
            "weight magnitude": total_weigth,
            "learning rate": self.learning_rate_var.numpy(),
        }

    def update_config(self, new_config):
        keys = ["learning_rate", "gamma", "episilon", "lambda"]
        for key in keys:
            if key not in new_config:
                return False, f"Missing {key} key"
        learning_rate = float(new_config["learning_rate"])
        self.learning_rate_var.assign(learning_rate)

        self.gamma = float(new_config["gamma"])
        self.episilon = float(new_config["episilon"])
        self.lamb = float(new_config["lambda"])

        return True, "ok"

    def current_config(self):
        return {
            "learning_rate": float(self.learning_rate_var.numpy()),
            "gamma": self.gamma,
            "episilon": self.episilon,
            "lambda": self.lamb,
        }
