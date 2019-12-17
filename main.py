#!/usr/bin/env python3

import argparse
import json
import time
import gzip
import sys


import tetris_gui
import train


def view(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("filename", help="Filename to visualize")
    parser.add_argument("--fps", type=float, default=60, help="Target fps")
    args = parser.parse_args(argv)

    fps = args.fps

    with gzip.open(args.filename, "rt") as f:
        states = json.load(f)

    print(f'Loaded {len(states)} states')
    print(f'Playback time: {len(states)/fps/60:.2f} minutes')

    start = time.time()
    gui = tetris_gui.TetrisGUI()
    for i, state in enumerate(states):
        gui.render(state)
        target_timestamp = start + i / fps
        remaining = target_timestamp - time.time()
        if remaining > 0:
            time.sleep(remaining)


def main():
    commands = {"train": train.train, "view": view}
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
