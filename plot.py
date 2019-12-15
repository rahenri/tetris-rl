#!/usr/bin/env python3

import argparse
import json
import os

import matplotlib.pyplot as plt
import mplcursors
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("experiments", nargs="+")
    parser.add_argument(
        "--experiment-dir",
        default="experiments",
        type=str,
        help="Directory containing experiments",
    )
    args = parser.parse_args()

    for experiment in args.experiments:
        filename = os.path.join(args.experiment_dir, experiment, "episodes.txt")
        episode_numbers = []
        pieces = []
        with open(filename) as f:
            for line in f:
                record = json.loads(line)
                episode_numbers.append(record["episode"])
                pieces.append(record["dropped_pieces"])
        plt.plot(episode_numbers, np.log(pieces), label=experiment)
    plt.legend()

    mplcursors.cursor(hover=True)
    plt.show()


if __name__ == "__main__":
    main()
