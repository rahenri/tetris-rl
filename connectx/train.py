#!/usr/bin/env python3

import argparse
import time
import os
import json
import resource
import sys
import multiprocessing
import shutil
import queue
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


def run_episode(agent, verbose, max_steps, board_shape, k, randomize):
    env = Env(board_shape, k)

    transitions = []

    for _ in range(max_steps):
        if verbose:
            print("env step" + "-" * 80)
            print(env)

        obs = env.obs()
        action, action_score = agent.act(env, randomize=randomize, verbose=verbose)

        if verbose:
            print(f"action: {action} score:{action_score}")

        reward, done = env.step(action)
        next_obs = env.obs()

        if done:
            next_obs = None

        transitions.append((obs, next_obs, reward))

        if done:
            break

    return transitions


def run_evaluation_episode(
    agent, enemy_cls, verbose, max_steps, enemy_first, board_shape, k, randomize
):
    env = Env(board_shape, k)

    total_reward = 0
    steps = 0

    enemy = enemy_cls()

    if enemy_first:
        _rew, done = env.step(enemy.act(env.board))
        # game won't end in one move
        assert not done

    for _ in range(max_steps):
        if verbose:
            print("env step" + "-" * 80)
            print(env)

        action, action_score = agent.act(env, randomize=randomize, verbose=verbose)

        if verbose:
            print(f"action: {action} score:{action_score}")

        reward, done = env.step(action)
        total_reward += reward
        steps += 1

        if done:
            break

        reward, done = env.step(enemy.act(env.board))
        total_reward -= reward
        if done:
            break

    if verbose:
        print("env over" + "-" * 80)
        print(env)
        print(f"Reward: {total_reward}")

    return total_reward, steps


def wait_for_filename(filename):
    while not os.path.exists(filename):
        time.sleep(0.1)


def run_rollout_worker(model_path, episode_queue, board_shape, k, model_version):
    print(random.random())
    agent = NNAgent(board_shape)
    counter = 0

    last_demo = time.time()

    wait_for_filename(model_path)
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
        transitions = run_episode(agent, demo, 10000, board_shape, k, not demo)
        episode_queue.put(transitions)
        counter += 1


def run_evaluator_worker(
    model_path,
    model_best_path,
    evaluator_queue,
    board_shape,
    k,
    model_version,
    agent_name,
):
    agent = NNAgent(board_shape)
    counter = 0

    wait_for_filename(model_path)

    last_version = model_version.value

    running_average = -1
    smoothing = 0.96

    best_average = -1
    best_version = 0

    if agent_name == "better-greedy":
        enemy_cls = agents.BetterGreedyAgent
    elif agent_name == "best-greedy":
        enemy_cls = agents.BestGreedyAgent
    else:
        raise RuntimeError(f"Unknown agent name {agent_name}")

    while True:
        next_version = model_version.value
        if next_version != last_version:
            agent.load_model(model_path)
            last_version = next_version
        all_rewards = []
        all_steps = []
        for _ in range(100):
            enemy_first = (counter % 2) == 1
            reward, steps = run_evaluation_episode(
                agent,
                agents.BetterGreedyAgent,
                False,
                10000,
                enemy_first,
                board_shape,
                k,
                False,
            )
            all_rewards.append(reward)
            all_steps.append(steps)
            counter += 1
        all_rewards = np.mean(all_rewards)
        all_steps = np.mean(all_steps)

        running_average = running_average * smoothing + all_rewards * (1.0 - smoothing)
        if running_average > best_average:
            best_average = running_average
            agent.save_model(model_best_path)
            best_version = last_version

        evaluator_queue.put(
            {
                "reward": all_rewards,
                "steps": all_steps,
                "running reward average": running_average,
                "best average": best_average,
                "best_version": best_version,
            }
        )


class Evaluator:
    def __init__(self, name):
        self.queue = multiprocessing.Queue(1)
        self.name = name
        self.last_values = {}

    def update(self):
        try:
            new_values = self.queue.get_nowait()
        except queue.Empty:
            return

        self.last_values = new_values


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
    model_tmp_path = os.path.join(model_dir, "model.tmp.npz")
    os.makedirs(model_dir, exist_ok=True)

    episode_queue = multiprocessing.Queue(8)
    model_version = 0
    model_version_shared = multiprocessing.Value("i", 0)
    for _ in range(4):
        worker = multiprocessing.Process(
            target=run_rollout_worker,
            args=(model_path, episode_queue, board_shape, k, model_version_shared),
        )
        worker.start()

    evaluators = []
    for evaluator_name, agent_name in [
        ("eval_better_greedy", "better-greedy"),
        ("eval_best_greedy", "best-greedy"),
    ]:
        evaluator = Evaluator(evaluator_name)
        model_best_path = os.path.join(model_dir, f"{evaluator_name}_best.npz")
        worker = multiprocessing.Process(
            target=run_evaluator_worker,
            args=(
                model_path,
                model_best_path,
                evaluator.queue,
                board_shape,
                k,
                model_version_shared,
                agent_name,
            ),
        )
        worker.start()
        evaluators.append(evaluator)

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

    batch_size = 1 << 12

    np.set_printoptions(threshold=100000)

    with open(os.path.join(output_dir, "episodes.txt"), "w") as output_log:
        try:
            for i_episode in range(episodes):
                start = time.time()
                config_manager.handle_events()

                filename = os.path.join(
                    output_dir, "ep_{:05d}.json.gz".format(i_episode)
                )

                start = time.time()
                for _ in range(30):
                    transitions = episode_queue.get()
                    for obs, next_obs, rew in transitions:
                        memory.add(obs, next_obs, rew)

                agent.save_model(model_tmp_path)
                os.rename(model_tmp_path, model_path)
                model_version += 1
                model_version_shared.value = model_version

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
                    "memory size": memory.size(),
                    "train step duration (ms)": learn_duration * 1000.0,
                    "rss (MB)": rss,
                }
                for key, val in train_metrics.items():
                    metrics[f"train/{key}"] = val

                for evaluator in evaluators:
                    evaluator.update()
                    for key, val in evaluator.last_values.items():
                        metrics[f"{evaluator.name}/{key}"] = val

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
