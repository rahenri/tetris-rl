#!/usr/bin/env python3

import argparse
import json
import logging
import os
import random
import shutil
import time
from os import path

from PIL import Image

from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout, Reshape
from keras.optimizers import Adam
from keras import backend as K

import numpy as np

from tetris_env import TetrisEnv
from board import Board, ACTION_MAP

print(K.tensorflow_backend._get_available_gpus())


def log_duration(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        logging.info("Function %s started", func.__qualname__)
        ret = func(*args, **kwargs)
        logging.info("Function %s took %fs", func.__qualname__, time.time() - start)
        return ret

    return wrapper


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
        self.cache = {}

        board_shape = state_shape[:2]
        self.board_shape = board_shape

        model = Sequential()
        model.add(Reshape(list(board_shape) + [1], input_shape=board_shape))
        model.add(Conv2D(32, 3, padding="same", activation="relu"))
        model.add(Conv2D(32, 3, padding="same", activation="relu"))
        model.add(MaxPooling2D())
        model.add(Dropout(0.5))
        model.add(Conv2D(64, 3, padding="same", activation="relu"))
        model.add(Conv2D(64, 3, padding="same", activation="relu"))
        model.add(MaxPooling2D())
        model.add(Dropout(0.5))
        model.add(Conv2D(128, 3, padding="same", activation="relu"))
        model.add(Conv2D(128, 3, padding="same", activation="relu"))
        model.add(MaxPooling2D())
        model.add(Dropout(0.5))
        model.add(Flatten())
        model.add(Dense(1024, activation="relu"))
        model.add(Dense(1024, activation="relu"))
        model.add(Dense(1, activation="linear"))
        model.compile(loss="mse", optimizer=Adam(lr=self.learning_rate))

        print(model.summary())

        self.value_model = model

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

        boards = np.array(boards)
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

    def _make_features(self, boards):
        return boards

    def _eval_many(self, boards):
        features = self._make_features(boards)
        out = self.value_model.predict(features).reshape([-1])
        assert len(out) == len(boards)
        return out

    @log_duration
    def learn(self, batch_size):
        if not self.history:
            return
        batch = random.sample(self.history, min(len(self.history), batch_size))
        features = np.array([entry.info["board"] for entry in batch])
        features_next = np.array(
            [
                entry.next_info["board"]
                if entry.next_info is not None
                else np.zeros(self.board_shape)
                for entry in batch
            ]
        )

        predictions = self._eval_many(features_next)
        target = np.zeros(predictions.shape)
        for i, pred in enumerate(predictions):
            nextq = pred if batch[i].next_info is not None else 0
            target[i] = batch[i].reward + self.gamma * nextq

        features = self._make_features(features)

        self.value_model.fit(
            features, target, epochs=1, verbose=0, batch_size=batch_size
        )


@log_duration
def run_episode(env, agent, demo):
    _, info = env.reset()

    total_reward = 0

    history = []

    for _ in range(10000000):
        action, action_score = agent.move(info, not demo)

        if demo:
            env.render()

        _, reward, done, next_info = env.step(ACTION_MAP[action])

        if demo:
            print("=" * 80)
            print(action, action_score)

        total_reward += reward

        if done:
            next_info = None

        history.append(PastState(info, reward))

        info = next_info

        if done:
            break

    agent.remember(history)

    return total_reward, history, env.pieces


def rmtree(dirpath):
    try:
        shutil.rmtree(dirpath)
    except FileNotFoundError:
        pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--episodes", default=100, type=int, help="Number of episodes to run"
    )
    parser.add_argument("--output-dir", type=str, help="Directory for output files")
    args = parser.parse_args()

    episodes = args.episodes

    output_dir = args.output_dir

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

    window = []

    last_summary = time.time()
    last_demo = time.time()

    best_window = 0
    best_episode = 0
    summary_every_sec = 1

    with open(path.join(output_dir, "episodes.txt"), "w") as output_log:
        try:
            for i_episode in range(episodes):
                now = time.time()
                demo = now - last_demo > 30

                total_reward, history, dropped_pieces = run_episode(env, agent, demo)

                is_best = False
                if dropped_pieces > best_episode:
                    best_episode = dropped_pieces
                    is_best = True

                window.append(dropped_pieces)
                if len(window) > 100:
                    window.pop(0)

                now = time.time()
                if now - last_summary > summary_every_sec:
                    m = np.mean(window)
                    best_window = max(best_window, m)
                    print(
                        "{}: Episode finished with reward {}"
                        ", moving average: {:.2f}, best average: {:.2f}, best episode: {}".format(
                            i_episode, dropped_pieces, m, best_window, best_episode
                        )
                    )
                    last_summary = time.time()

                output = {
                    "episode": i_episode,
                    "dropped_pieces": dropped_pieces,
                    "reward": total_reward,
                }
                output_log.write(json.dumps(output) + "\n")
                output_log.flush()

                if demo:
                    last_demo = time.time()

                if is_best:
                    d = path.join(output_dir, "ep_{:05d}".format(i_episode))
                    rmtree(d)
                    os.makedirs(d, exist_ok=True)
                    for i, s in enumerate(history):
                        info = s.info
                        img = env.render_info(info)
                        im = Image.fromarray(img)
                        im.save(path.join(d, "step_{:06d}.png".format(i)))

                agent.learn(1 << 14)

        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
