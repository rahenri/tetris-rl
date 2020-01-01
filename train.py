import argparse
import random
import time
import os
import gzip
import json
import resource

import numpy as np
import tensorflow as tf
from tabulate import tabulate
from memory_profiler import profile

from board import Board
from drop_wrapper import DropWrapper
from gui_wrapper import GuiWrapper
from memory import Memory
from model import AgentModel
from tetris_engine import Moves
from tetris_env import TetrisEnv
from tetris_gui import TetrisGUI

# from util import log_duration, log_memory_change, log_memory_change_context


# A start and reward pair
class PastState:
    def __init__(self, info, reward):
        self.info = info
        self.reward = reward

    def __repr__(self):
        return str((self.info["board"], self.reward))


# An agent that can lean and play tetris. It learns via reinforcement learning
# and its model is a multi convolutional neural layer net. The model predicts
# the total future reward from a given board configuration without regarding to
# curently active piece, so it is used to pick where to place each piece
# instead of using it to control the piece at every step. The later would be
# harder to train while the algorithm to control the piece is easy to do
# using more traditional methods, ie, BFS.
class NNAgent:
    def __init__(self, state_shape, action_size):
        self.learning_rate = 0.001
        self.state_shape = state_shape
        self.action_size = action_size
        self.gamma = 0.999
        self.episilon = 0.01
        self.lamb = 0.98

        board_shape = state_shape[:2]
        self.board_shape = board_shape

        self.value_model = AgentModel("value", board_shape)
        self.target_value_model = AgentModel("target_value", board_shape)

        self.value_model.build(input_shape=(None,) + board_shape)
        self.target_value_model.build(input_shape=(None,) + board_shape)

        for var, var_target in zip(
            self.value_model.trainable_variables,
            self.target_value_model.trainable_variables,
        ):
            var.assign(var_target)

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        self.loss_function = tf.keras.losses.MeanSquaredError()

    # Pick a move given the game state, the move is selected by listing out all
    # possible places to place the current piece and evaluating the resulting
    # board with the neural net, it returns the movement that leads to the
    # state which maximize the predicted future total reward.
    def act(self, info, randomize=False):
        x = info["piece_x"]
        y = info["piece_y"]
        shape = info["piece_shape"]
        rotation = info["piece_rotation"]

        board = info["board"]

        start = Board(board, shape, x, y, rotation)

        visited, finals = start.list_paths()

        if not finals:
            print(info)
            raise RuntimeError("No finals, this shouldn't happen")

        boards = []
        rewards = []
        for node in finals:
            boards.append(node.board)
            rewards.append(node.reward())

        boards = np.array(boards, dtype=np.float32)
        scores = self._eval_many(boards)

        best_end = None
        best_end_score = None
        if not randomize or random.random() > self.episilon:
            for i, node in enumerate(finals):
                score = rewards[i] + self.gamma * scores[i]
                if best_end is None or score > best_end_score:
                    best_end = node
                    best_end_score = score
        else:
            best = random.randint(0, len(boards) - 1)
            best_end = finals[best]
            best_end_score = scores[best]

        node = best_end
        actions = []
        while node != start:
            (action, next_node) = visited[node.tup()]
            if action != Moves.DROP:
                actions.append(action)
            node = next_node
        actions.reverse()

        return actions, best_end_score

    def _eval(self, board):
        return self._eval_many(np.array([board]))[0]

    def _eval_many(self, boards):
        boards = tf.convert_to_tensor(boards, dtype=np.float32)
        return self.value_model(boards).numpy().reshape(-1)

    def _make_features(self, memory, batch_size):
        boards, next_boards, rewards, dones = memory.sample(batch_size)

        boards = tf.convert_to_tensor(boards, dtype=np.float32)
        boards_next = tf.convert_to_tensor(next_boards, dtype=np.float32)
        not_dones = tf.convert_to_tensor(1 - dones, dtype=np.float32)
        rewards = tf.convert_to_tensor(rewards, dtype=np.float32)

        predictions = tf.reshape(self.target_value_model(boards_next), [-1])
        targets = predictions * self.gamma * not_dones + rewards

        return boards, targets

    def trainable_variables(self):
        return (
            self.value_model.trainable_variables
            + self.target_value_model.trainable_variables
        )

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
            predicions = self.value_model(features, training=True)
            loss = self.loss_function(target, predicions)
        gradients = tape.gradient(loss, self.value_model.trainable_variables)
        self.optimizer.apply_gradients(
            zip(gradients, self.value_model.trainable_variables)
        )
        for a, b in zip(
            self.value_model.trainable_variables,
            self.target_value_model.trainable_variables,
        ):
            b.assign(b * self.lamb + a * (1.0 - self.lamb))
        del tape
        return loss.numpy()


def run_episode(env, agent, demo, memory, max_steps):
    _, info = env.reset()

    total_reward = 0
    steps = 0

    for _ in range(max_steps):
        action, _action_score = agent.act(info, not demo)

        _, reward, done, next_info = env.step(action)
        total_reward += reward
        steps += 1

        if done:
            next_info = None

        if memory is not None:
            memory.add(info, next_info, reward)

        info = next_info

        if done:
            break

    return total_reward, steps


def make_env(gui, record=True):
    env = TetrisEnv()

    if gui is not None:
        env = GuiWrapper(env, gui)

    if record:
        env = RecorderWrapper(env)
        recorder = env
    else:
        recorder = None

    env = DropWrapper(env)

    return env, recorder


class RecorderWrapper:
    def __init__(self, env):
        self.env = env
        self.action_space = env.action_space
        self.observation_space = env.observation_space
        self.output = None

    def start_episode(self, filename):
        if self.output:
            self.output.close()
        self.output = gzip.open(filename, "wt")

    def end_episode(self):
        if self.output:
            self.output.close()
            self.output = None

    def _store(self, info):
        board = info["color_board_with_falling_piece"]
        board = "".join(["".join([str(item) for item in r]) for r in board])
        self.output.write(board + "\n")

    def reset(self):
        state, info = self.env.reset()
        self._store(info)
        return state, info

    def step(self, action):
        state, reward, done, info = self.env.step(action)
        self._store(info)
        return state, reward, done, info

    @property
    def pieces(self):
        return self.env.pieces


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

    return train_real(
        args.episodes, args.name, args.experiment_dir, args.load_model, args.gui
    )


def train_real(episodes, name, experiment_dir, load_model, enable_gui):

    output_dir = os.path.join(experiment_dir, name)

    if os.path.exists(output_dir):
        print(f"An experiment named {name} already exists")
        return

    os.makedirs(output_dir, exist_ok=True)

    print(f"Will run a total of {episodes} episodes")
    print(f"Writting files to {output_dir}")

    gui = None
    if enable_gui:
        gui = TetrisGUI()

    env, recorder = make_env(gui)

    # logging.basicConfig(
    #     format='%(asctime)s %(filename)s:%(lineno)d %(message)s',
    #     level=logging.INFO, datefmt='%Y/%m/%d %H:%M:%S')

    state_shape = env.observation_space.shape
    action_size = env.action_space.n

    print(f"State shape: {state_shape}")
    print(f"Action size: {action_size}")

    agent = NNAgent(state_shape, action_size)

    memory = Memory(10000000)

    if load_model:
        filename = os.path.join(
            experiment_dir, load_model, "model_snapshots", "model.npz"
        )
        print(f"Loading model from {filename}")
        agent.load_model(filename)

    last_demo = time.time()

    best_episode_steps = 0
    best_episode_reward = 0

    batch_size = 1 << 12
    max_steps = 1000

    model_dir = os.path.join(output_dir, "model_snapshots")
    os.makedirs(model_dir, exist_ok=True)

    with open(os.path.join(output_dir, "episodes.txt"), "w") as output_log:
        try:
            for i_episode in range(episodes):
                now = time.time()
                demo = now - last_demo > 30

                filename = os.path.join(
                    output_dir, "ep_{:05d}.json.gz".format(i_episode)
                )
                recorder.start_episode(filename)

                start = time.time()
                total_reward, steps = run_episode(env, agent, demo, memory, max_steps)
                episode_duration_per_step = (time.time() - start) / steps

                recorder.end_episode()
                dropped_pieces = env.pieces

                is_best = False
                if dropped_pieces > best_episode_steps:
                    best_episode_steps = dropped_pieces
                if total_reward > best_episode_reward:
                    best_episode_reward = total_reward
                    is_best = True


                if demo:
                    last_demo = time.time()

                if is_best:
                    agent.save_model(os.path.join(model_dir, "model.npz"))

                start_learn = time.time()
                if memory.size() >= 1024:
                    loss = agent.train(memory, batch_size)
                else:
                    loss = 0
                learn_duration = time.time() - start_learn
                rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024

                metrics = {
                    "episode": i_episode,
                    "dropped_pieces": dropped_pieces,
                    "reward": total_reward,
                    "loss": float(loss),
                    "best episode (steps)": best_episode_steps,
                    "best episode (reward)": best_episode_reward,
                    "memory size": memory.size(),
                    "train step duration (ms)": learn_duration * 1000.0,
                    "time per env step(ms)": episode_duration_per_step * 1000.0,
                    "rss (MB)": rss,
                }
                print(
                    tabulate(
                        metrics.items(), tablefmt="psql", headers=["name", "value"]
                    )
                )
                output_log.write(json.dumps(metrics) + "\n")
                output_log.flush()

        except KeyboardInterrupt:
            pass
