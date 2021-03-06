#!/usr/bin/env python3

import argparse
import json
import time
import gzip
import sys
import os


import tetris_gui
import train
from tetris_env import TetrisEnv
from tetris_gui import TetrisGUI


def view(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("filename", help="Filename to visualize")
    parser.add_argument("--fps", type=float, default=60, help="Target fps")
    args = parser.parse_args(argv)

    fps = args.fps

    with gzip.open(args.filename, "rt") as f:
        states = []
        for line in f:
            states.append(line.strip())

    print(f"Loaded {len(states)} states")
    print(f"Playback time: {len(states)/fps/60:.2f} minutes")

    start = time.time()
    gui = tetris_gui.TetrisGUI()
    for i, state in enumerate(states):

        rows = []
        while state:
            r = state[:24]
            state = state[24:]
            row = []
            for l in r:
                if l != ".":
                    l = int(l)
                row.append(l)
            rows.append(row)

        gui.render(rows)
        target_timestamp = start + i / fps
        if not gui.process_events():
            break
        remaining = target_timestamp - time.time()
        if remaining > 0:
            time.sleep(remaining)


def rollout(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--load-model", required=True, help="Model weights")
    args = parser.parse_args(argv)

    gui = TetrisGUI()
    env, _ = train.make_env(gui, record=False)

    state_shape = env.observation_space.shape
    action_size = env.action_space.n

    agent = train.NNAgent(state_shape, action_size)

    agent.load_model(args.load_model)

    while True:
        rew, steps = train.run_episode(env, agent, True, None, 100000)
        print(f"reward: {rew} steps: {steps}")


def main():
    commands = {"train": train.train, "view": view, "rollout": rollout}
    name = sys.argv[1]

    command = commands.get(name, None)
    if command is None:
        print(f"Invalid command {name}, available commands are:")
        for n in commands:
            print(f"  {n}")
        return

    command(sys.argv[2:])


if __name__ == "__main__":
    main()
