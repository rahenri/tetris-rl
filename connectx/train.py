#!/usr/bin/env python3

import argparse
import random
import time
import os
import json
import resource
import sys

import numpy as np
import tensorflow as tf

from board import Board
from agents import RandomAgent
from memory import Memory
from model import Model
from tabulate import tabulate


# A start and reward pair
class PastState:
    def __init__(self, info, reward):
        self.info = info
        self.reward = reward

    def __repr__(self):
        return str((self.info["board"], self.reward))


class NNAgent:
    def __init__(self, state_shape, action_size):
        self.learning_rate = 0.001
        self.state_shape = state_shape
        self.action_size = action_size
        self.gamma = 1.0
        self.episilon = 0.01
        self.lamb = 0.99

        board_shape = state_shape[:2]
        self.board_shape = board_shape

        self.value_net = Model("value", board_shape, 1)
        self.target_net = Model("target", board_shape, 1)

        self.value_net.build(input_shape=(None,) + board_shape)
        self.target_net.build(input_shape=(None,) + board_shape)

        for var, var_target in zip(
            self.value_net.trainable_variables, self.target_net.trainable_variables,
        ):
            var.assign(var_target)

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        self.loss_function = tf.keras.losses.MeanSquaredError()

    def act(self, board, randomize=False):
        actions = board.list_moves()
        assert len(actions) > 0

        if not randomize or random.random() < self.episilon:
            return random.choice(actions), 0

        boards = []
        for a in actions:
            boards.append(board.cells_if_move(a))

        obs = tf.convert_to_tensor(boards, tf.float32)
        scores = self.value_net(obs).numpy().reshape(-1)

        best_score = None
        best_action = None
        for action, score in zip(actions, scores):
            if best_score is None or score > best_score:
                best_score = score
                best_action = action

        return best_action, best_score

    def _make_features(self, memory, batch_size):
        boards, next_boards, rewards, dones = memory.sample(batch_size)

        boards = tf.convert_to_tensor(boards, dtype=np.float32)
        boards_next = tf.convert_to_tensor(next_boards, dtype=np.float32)
        not_dones = tf.convert_to_tensor(1 - dones, dtype=np.float32)
        rewards = tf.convert_to_tensor(rewards, dtype=np.float32)

        predictions = tf.reshape(self.target_net(boards_next), [-1])
        targets = predictions * self.gamma * not_dones + rewards

        return boards, targets

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
        features, target = self._make_features(memory, batch_size)

        with tf.GradientTape() as tape:
            predicions = self.value_net(features, training=True)
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
        return {
            "loss": loss.numpy(),
            "mean target value": target.numpy().mean(),
        }


class Env:
    def __init__(self, enemy_first):
        self.board = Board(6, 7, 4)
        self.enemy = RandomAgent()
        if enemy_first:
            self.board.act(self.enemy.act(self.board))
        self.player = self.board.turn

    def _rew(self):
        if not self.board.finished():
            return 0
        winner = self.board.winner()
        if winner is None:
            return 0
        if winner == self.player:
            return 1
        return -1

    def obs(self):
        return self.board.cells()

    def step(self, action):
        ended = self.board.act(action)
        if ended:
            return self._rew(), True
        enemy_action = self.enemy.act(self.board)
        ended = self.board.act(enemy_action)
        return self._rew(), ended


def run_episode(agent, demo, memory, max_steps, enemy_first):
    env = Env(enemy_first)
    obs = env.obs()

    total_reward = 0
    steps = 0

    for _ in range(max_steps):
        action, action_score = agent.act(env.board, not demo)

        reward, done = env.step(action)
        next_obs = env.obs()
        total_reward += reward
        steps += 1

        if done:
            next_obs = None

        if memory is not None:
            memory.add(obs, next_obs, reward)

        obs = next_obs

        if done:
            break

    return total_reward, steps


def train(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--episodes", default=100, type=int, help="Number of episodes to run"
    )
    parser.add_argument(
        "--name", type=str, help="Name of the experiment", required=True
    )
    parser.add_argument(
        "--experiment-dir",
        default="experiments",
        type=str,
        help="Directory containing experiments",
    )
    parser.add_argument(
        "--load-model",
        default=None,
        type=str,
        help="Experiment name to load model from",
    )
    parser.add_argument(
        "--gui", default=False, help="Whether to show a gui", action="store_true"
    )
    args = parser.parse_args(argv)

    return train_real(args.episodes, args.name, args.experiment_dir, args.load_model)


def train_real(episodes, name, experiment_dir, load_model):

    output_dir = os.path.join(experiment_dir, name)

    os.makedirs(output_dir, exist_ok=True)

    print(f"Will run a total of {episodes} episodes")
    print(f"Writting files to {output_dir}")

    # logging.basicConfig(
    #     format='%(asctime)s %(filename)s:%(lineno)d %(message)s',
    #     level=logging.INFO, datefmt='%Y/%m/%d %H:%M:%S')

    agent = NNAgent((6, 7), 7)

    memory = Memory(10000000)

    if load_model:
        filename = os.path.join(
            experiment_dir, load_model, "model_snapshots", "model.npz"
        )
        print(f"Loading model from {filename}")
        agent.load_model(filename)

    last_demo = time.time()

    best_episode_reward = 0
    best_episode = 0

    batch_size = 1 << 12
    max_steps = 1000

    model_dir = os.path.join(output_dir, "model_snapshots")
    os.makedirs(model_dir, exist_ok=True)

    reward_average = 0

    with open(os.path.join(output_dir, "episodes.txt"), "w") as output_log:
        try:
            for i_episode in range(episodes):
                now = time.time()
                demo = now - last_demo > 30

                filename = os.path.join(
                    output_dir, "ep_{:05d}.json.gz".format(i_episode)
                )

                want_steps = 100
                while want_steps > 0:
                    start = time.time()
                    total_reward, steps = run_episode(
                        agent, demo, memory, max_steps, i_episode % 2
                    )
                    episode_duration_per_step = (time.time() - start) / steps
                    want_steps -= steps

                is_best = False
                if total_reward > best_episode_reward:
                    best_episode_reward = total_reward
                    best_episode = i_episode
                    is_best = True

                if demo:
                    last_demo = time.time()

                if is_best:
                    agent.save_model(os.path.join(model_dir, "model.npz"))

                start_learn = time.time()
                if memory.size() >= 1024:
                    train_metrics = agent.train(memory, batch_size)
                else:
                    train_metrics = {}
                learn_duration = time.time() - start_learn
                rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024

                a = 0.96
                reward_average = reward_average * a + total_reward * (1 - a)

                metrics = {
                    "episode": i_episode,
                    "steps": steps,
                    "reward": total_reward,
                    "reward_average": reward_average,
                    "best episode reward": best_episode_reward,
                    "best episode number": best_episode,
                    "memory size": memory.size(),
                    "train step duration (ms)": learn_duration * 1000.0,
                    "time per env step(ms)": episode_duration_per_step * 1000.0,
                    "rss (MB)": rss,
                }
                for k, v in train_metrics.items():
                    metrics["train/" + k] = v

                for k, v in metrics.items():
                    metrics[k] = str(v)
                print(
                    tabulate(
                        metrics.items(), tablefmt="psql", headers=["name", "value"]
                    )
                )
                output_log.write(json.dumps(metrics) + "\n")
                output_log.flush()

        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    train(sys.argv[1:])
