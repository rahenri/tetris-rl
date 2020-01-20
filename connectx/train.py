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
from tabulate import tabulate

from board import Board, Position
from agents import BetterGreedyAgent
from memory import Memory
from model import Model
from observation import SingleObservation, ObservationVector


class NNAgent:
    def __init__(self, board_shape):
        self.learning_rate = 0.001
        self.board_shape = board_shape
        self.gamma = 0.95
        self.episilon = 0.1
        self.lamb = 0.98

        self.board_shape = board_shape

        self.value_net = Model("value", board_shape)
        self.target_net = Model("target", board_shape)

        for var, var_target in zip(
            self.value_net.trainable_variables, self.target_net.trainable_variables,
        ):
            var.assign(var_target)

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
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

        sample_score = self.target_net(sample_obs).numpy().reshape(-1)[0]

        return {
            "loss": loss.numpy(),
            "mean target value": target.numpy().mean(),
            "sample score": sample_score,
        }


class Env:
    def __init__(self, board_shape, k):
        self.board_shape = board_shape
        self.k = k
        height, width = board_shape
        self.board = Board(height, width, k)

    def copy(self):
        out = Env(self.board_shape, self.k)
        out.board = self.board.copy()
        return out

    def obs(self):
        return SingleObservation(self.board.cells(), self.board.turn() - 1)

    def would_win(self, action, player):
        row = self.board.row(action)
        pos = Position(row, action)
        return self.board.would_win(pos, player)

    def step(self, action):
        ended = self.board.step(action)
        rew = 1 if ended and (self.board.winner() is not None) else 0
        return rew, ended

    def turn(self):
        return self.board.turn()

    def __repr__(self):
        return repr(self.board)


def run_episode(agent, demo, memory, max_steps, enemy_first, board_shape, k):
    env = Env(board_shape, k)

    total_reward = 0
    steps = 0

    enemy = BetterGreedyAgent()

    if enemy_first:
        _rew, done = env.step(enemy.act(env.board))
        # game won't end in one move
        assert not done

    for _ in range(max_steps):
        if demo:
            print("env step" + "-" * 80)
            print(env)

        obs = env.obs()
        action, action_score = agent.act(env, randomize=not demo, verbose=demo)

        if demo:
            print(f"action: {action} score:{action_score}")

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

        rew, done = env.step(enemy.act(env.board))
        total_reward -= rew
        if done:
            break

    if demo:
        print("env over" + "-" * 80)
        print(env)
        print(f"Reward: {total_reward}")

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

    board_shape = (6, 7)
    k = 4

    agent = NNAgent(board_shape)

    memory = Memory(board_shape, 10000000)

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

    np.set_printoptions(threshold=100000)
    counter = 0

    with open(os.path.join(output_dir, "episodes.txt"), "w") as output_log:
        try:
            for i_episode in range(episodes):
                now = time.time()
                demo = now - last_demo > 30
                if demo:
                    last_demo = time.time()

                filename = os.path.join(
                    output_dir, "ep_{:05d}.json.gz".format(i_episode)
                )

                total_rewards = []
                steps = []
                start = time.time()
                # demo = True
                for _ in range(15):
                    counter += 1
                    enemy_first = counter % 2 == 1
                    rew, step = run_episode(
                        agent, demo, memory, max_steps, enemy_first, board_shape, k
                    )
                    total_rewards.append(rew)
                    steps.append(step)
                    demo = False

                episode_duration_per_step = (time.time() - start) / np.sum(steps)

                total_reward = np.mean(total_rewards)
                steps = np.mean(steps)

                a = 0.96
                reward_average = reward_average * a + total_reward * (1 - a)

                is_best = False
                if i_episode > 100 and reward_average > best_episode_reward:
                    best_episode_reward = reward_average
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
                for key, val in train_metrics.items():
                    metrics["train/" + key] = val

                for key, val in metrics.items():
                    metrics[key] = str(val)
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
