#!/usr/bin/env python3

import argparse
import time
import os
import json
import resource
import sys
import multiprocessing
import shutil
import random

import numpy as np
from tabulate import tabulate

from board import Board, Position
import agents
from memory import Memory
from observation import SingleObservation
from nnagent import NNAgent
from config_manager import ConfigManager
import webui


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


def run_episode(agent, demo, max_steps, enemy_first, board_shape, k):
    env = Env(board_shape, k)

    total_reward = 0
    steps = 0

    enemy = agents.BetterGreedyAgent()

    if enemy_first:
        _rew, done = env.step(enemy.act(env.board))
        # game won't end in one move
        assert not done

    transitions = []

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

        transitions.append((obs, next_obs, reward))

        if done:
            break

        reward, done = env.step(enemy.act(env.board))
        total_reward -= reward
        if done:
            break

    if demo:
        print("env over" + "-" * 80)
        print(env)
        print(f"Reward: {total_reward}")

    return total_reward, steps, transitions


def run_worker(model_path, episode_queue, board_shape, k, model_version):
    agent = NNAgent(board_shape)
    counter = 0

    last_demo = time.time()

    while not os.path.exists(model_path):
        time.sleep(0.1)

    last_version = model_version.value

    while True:
        demo = False
        next_version = model_version.value
        if next_version != last_version:
            agent.load_model(model_path)
            last_version = next_version
        now = time.time()
        if now - last_demo > 30:
            demo = True
            last_demo = now
        enemy_first = (counter % 2) == 1
        rew, step, transitions = run_episode(
            agent, demo, 10000, enemy_first, board_shape, k
        )
        episode_queue.put((rew, step, transitions))
        counter += 1


def train(episodes, name, experiment_dir, load_model):

    output_dir = os.path.join(experiment_dir, name)

    os.makedirs(output_dir, exist_ok=True)

    print(f"Will run a total of {episodes} episodes")
    print(f"Writting files to {output_dir}")

    # logging.basicConfig(
    #     format='%(asctime)s %(filename)s:%(lineno)d %(message)s',
    #     level=logging.INFO, datefmt='%Y/%m/%d %H:%M:%S')

    board_shape = (6, 7)
    k = 4

    model_dir = os.path.join(output_dir, "model_snapshots")
    model_path = os.path.join(model_dir, "model.npz")
    model_best_path = os.path.join(model_dir, "model_best.npz")
    model_tmp_path = os.path.join(model_dir, "model.tmp.npz")
    os.makedirs(model_dir, exist_ok=True)

    episode_queue = multiprocessing.Queue(1)
    model_version = 0
    model_version_shared = multiprocessing.Value("i", 0)
    for _ in range(4):
        worker = multiprocessing.Process(
            target=run_worker,
            args=(model_path, episode_queue, board_shape, k, model_version_shared),
        )
        worker.start()

    agent = NNAgent(board_shape)
    agent.save_model(model_path)

    config_manager = ConfigManager(agent.update_config, agent.current_config())
    webui.run_http_server(config_manager)

    memory = Memory(board_shape, 100000000)

    if load_model:
        filename = os.path.join(
            experiment_dir, load_model, "model_snapshots", "model.npz"
        )
        print(f"Loading model from {filename}")
        agent.load_model(filename)

    best_episode_reward = -1e9
    best_episode = 0

    batch_size = 1 << 12

    reward_average = -1

    np.set_printoptions(threshold=100000)
    smoothing = 0.96

    with open(os.path.join(output_dir, "episodes.txt"), "w") as output_log:
        try:
            for i_episode in range(episodes):
                start = time.time()
                config_manager.handle_events()

                filename = os.path.join(
                    output_dir, "ep_{:05d}.json.gz".format(i_episode)
                )

                total_rewards = []
                steps = []
                start = time.time()
                for _ in range(30):
                    rew, step, transitions = episode_queue.get()
                    total_rewards.append(rew)
                    steps.append(step)
                    for obs, next_obs, rew in transitions:
                        memory.add(obs, next_obs, rew)

                episode_duration_per_step = (time.time() - start) / np.sum(steps)

                total_reward = np.mean(total_rewards)
                steps = np.mean(steps)

                reward_average += (1 - smoothing) * (total_reward - reward_average)

                agent.save_model(model_tmp_path)
                os.rename(model_tmp_path, model_path)
                model_version += 1
                model_version_shared.value = model_version

                if i_episode > 100 and reward_average > best_episode_reward:
                    best_episode_reward = reward_average
                    best_episode = i_episode
                    shutil.copyfile(model_path, model_best_path)

                start_learn = time.time()
                if memory.size() >= 1024:
                    train_metrics = agent.train(memory, batch_size)
                else:
                    train_metrics = {}
                learn_duration = time.time() - start_learn
                rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024

                duration = time.time() - start

                metrics = {
                    "episode": i_episode,
                    "iteration duration": duration,
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


def main(argv):
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

    return train(args.episodes, args.name, args.experiment_dir, args.load_model)


if __name__ == "__main__":
    main(sys.argv[1:])
