#!/usr/bin/env python3


import argparse
import matplotlib.pyplot as plt
import json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("files", nargs="+")
    args = parser.parse_args()

    for filename in args.files:
        episode_numbers = []
        pieces = []
        with open(filename) as f:
            for line in f:
                record = json.loads(line)
                episode_numbers.append(record["episode"])
                pieces.append(record["dropped_pieces"])
        plt.plot(episode_numbers, pieces)
        plt.title(filename)
    plt.show()


if __name__ == "__main__":
    main()
