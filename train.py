import argparse
import random
import time
import os
import gzip
import json

import numpy as np
import tensorflow as tf

from board import Board, ACTION_MAP
from tetris_env import TetrisEnv
from util import log_duration
from model import AgentModel
from tetris_gui import TetrisGUI

# Represents an entry in the agent's memory, contains a state, the following
# state and the reward received in the transition
class MemoryEntry:
    def __init__(self, info, next_info, reward):
        self.info = info
        self.next_info = next_info
        self.reward = reward

    def __repr__(self):
        return str((self.info["board"], self.reward))


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
        self.cache = {}

        board_shape = state_shape[:2]
        self.board_shape = board_shape

        self.value_model = AgentModel("value", board_shape)
        self.target_value_model = AgentModel("target_value", board_shape)

        self.value_model.build(input_shape=(None,) + board_shape)
        self.target_value_model.build(input_shape=(None,) + board_shape)

        for a, b in zip(
            self.value_model.trainable_variables,
            self.target_value_model.trainable_variables,
        ):
            a.assign(b)

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        self.loss_function = tf.keras.losses.MeanSquaredError()

        self.history = []

    # Add a full game to its memory, a game is a sequence of state-reward pairs.
    def remember(self, history: [PastState]):
        # Why do I filter zero reward stuff?
        filtered_history = []
        for record in history:
            if record.reward == 0:
                continue
            filtered_history.append(record)
        history = filtered_history

        for i, record in enumerate(history):
            next_i = i + 1
            next_info = history[next_i].info if next_i < len(history) else None

            self.history.append(MemoryEntry(record.info, next_info, record.reward))

    # Pick a move given the game state, the move is selected by listing out all
    # possible places to place the current piece and evaluating the resulting
    # board with the neural net, it returns the movement that leads to the
    # state which maximize the predicted future total reward.
    def move(self, info, randomize=False):
        if "piece_shape" not in info:
            return "DOWN", self._eval(info["board"])

        x = info["piece_x"]
        y = info["piece_y"]
        shape = info["piece_shape"]
        rotation = info["piece_rotation"]

        board = info["board"]

        start = Board(board, shape, x, y, rotation)
        h = start.full_hash()
        if h in self.cache:
            action, reward = self.cache[h]
            return action, reward
        self.cache = {}

        visited, finals = start.list_paths()

        if not finals:
            print(info)
            raise RuntimeError("No finals")

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
        actions = ["DOWN"]
        while node != start:
            (action, n) = visited[node.tup()]
            actions.append(action)
            node = n
            self.cache[n.full_hash()] = (action, best_end_score)
        action = actions[-1]

        return action, best_end_score

    def _eval(self, board):
        return self._eval_many(np.array([board]))[0]

    def _eval_many(self, boards):
        boards = tf.convert_to_tensor(boards, dtype=np.float32)
        return self.value_model(boards).numpy().reshape(-1)

    def _make_features(self, batch_size):
        batch = random.sample(self.history, min(len(self.history), batch_size))
        features = tf.convert_to_tensor(
            [entry.info["board"] for entry in batch], dtype=np.float32
        )
        zeros = np.zeros(self.board_shape)
        features_next = tf.convert_to_tensor(
            [
                entry.next_info["board"] if entry.next_info is not None else zeros
                for entry in batch
            ],
            dtype=np.float32,
        )

        predictions = self.target_value_model(features_next).numpy().reshape(-1)
        target = np.zeros(predictions.shape)
        for i, pred in enumerate(predictions):
            nextq = pred if batch[i].next_info is not None else 0
            target[i] = batch[i].reward + self.gamma * nextq
        target = tf.convert_to_tensor(target.reshape([-1, 1]), dtype=np.float32)

        return features, target

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

    @log_duration
    def learn(self, batch_size):
        if not self.history:
            return
        features, target = self._make_features(batch_size)

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
        return loss.numpy()


@log_duration
def run_episode(env, agent, demo, gui):
    _, info = env.reset()

    total_reward = 0

    history = []

    for _ in range(10000000):
        action, _action_score = agent.move(info, not demo)

        if demo and gui:
            gui.render(info["color_board_with_falling_piece"])

        _, reward, done, next_info = env.step(ACTION_MAP[action])

        total_reward += reward

        if done:
            next_info = None

        history.append(PastState(info, reward))

        info = next_info

        if done:
            break

    return total_reward, history, env.pieces


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

    episodes = args.episodes

    output_dir = os.path.join(args.experiment_dir, args.name)

    if os.path.exists(output_dir):
        print(f"An experiment named {args.name} already exists")
        return

    os.makedirs(output_dir, exist_ok=True)

    print(f"Will run a total of {episodes} episodes")
    print(f"Writting files to {output_dir}")

    env = TetrisEnv()

    # logging.basicConfig(
    #     format='%(asctime)s %(filename)s:%(lineno)d %(message)s',
    #     level=logging.INFO, datefmt='%Y/%m/%d %H:%M:%S')

    state_shape = env.observation_space.shape
    action_size = env.action_space.n

    print(f"State shape: {state_shape}")
    print(f"Action size: {action_size}")

    agent = NNAgent(state_shape, action_size)

    if args.load_model:
        filename = os.path.join(
            args.experiment_dir, args.load_model, "model_snapshots", "model.npz"
        )
        print(f"Loaging model from {filename}")
        agent.load_model(filename)

    window = []

    last_summary = time.time()
    last_demo = time.time()

    best_window = 0
    best_episode = 0
    summary_every_sec = 1

    model_dir = os.path.join(output_dir, "model_snapshots")
    os.makedirs(model_dir, exist_ok=True)

    gui = None
    if args.gui:
        gui = TetrisGUI()

    with open(os.path.join(output_dir, "episodes.txt"), "w") as output_log:
        try:
            for i_episode in range(episodes):
                now = time.time()
                demo = now - last_demo > 30

                total_reward, history, dropped_pieces = run_episode(
                    env, agent, demo, gui
                )
                agent.remember(history)

                is_best = False
                if dropped_pieces > best_episode:
                    best_episode = dropped_pieces
                    is_best = True

                window.append(dropped_pieces)
                if len(window) > 100:
                    window.pop(0)

                if demo:
                    last_demo = time.time()

                if is_best:
                    agent.save_model(os.path.join(model_dir, "model.npz"))

                # save episode
                d = os.path.join(output_dir, "ep_{:05d}.json.gz".format(i_episode))
                with gzip.open(d, "wt") as f:
                    json.dump(
                        [h.info["color_board_with_falling_piece"] for h in history], f
                    )

                loss = agent.learn(1 << 14)

                now = time.time()
                if now - last_summary > summary_every_sec:
                    m = np.mean(window)
                    best_window = max(best_window, m)
                    print(
                        "{}: Episode finished with reward {}"
                        ", moving average: {:.2f}, best average: {:.2f}, best episode: {}, loss: {}".format(
                            i_episode,
                            dropped_pieces,
                            m,
                            best_window,
                            best_episode,
                            loss,
                        )
                    )
                    last_summary = time.time()

                output = {
                    "episode": i_episode,
                    "dropped_pieces": dropped_pieces,
                    "reward": total_reward,
                    "loss": float(loss),
                }
                output_log.write(json.dumps(output) + "\n")
                output_log.flush()

        except KeyboardInterrupt:
            pass
