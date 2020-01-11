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
    metric = "reward_average"
    num_buckets = 200

    for experiment in args.experiments:
        filename = os.path.join(args.experiment_dir, experiment, "episodes.txt")
        episode_numbers = []
        values = []
        with open(filename) as f:
            for line in f:
                record = json.loads(line)
                episode_numbers.append(int(record["episode"]))
                values.append(float(record[metric]))
        minx = np.min(episode_numbers)
        maxx = np.max(episode_numbers)

        bucket_max = [None] * num_buckets
        bucket_min = [None] * num_buckets
        bucket_sum = [0] * num_buckets
        bucket_count = [0] * num_buckets

        bucket_x = []
        for i in range(num_buckets):
            bucket_x.append(minx + (maxx - minx) * i / num_buckets)

        for i, v in enumerate(values):
            x = min(int((episode_numbers[i] - minx) / (maxx - minx) * num_buckets), num_buckets - 1)

            acc = bucket_max[x]
            if acc is not None:
                acc = max(acc, v)
            else:
                acc = v
            bucket_max[x] = acc

            acc = bucket_min[x]
            if acc is not None:
                acc = min(acc, v)
            else:
                acc = v
            bucket_min[x] = acc

            bucket_sum[x] += v
            bucket_count[x] += 1

        bucket_mean = np.array(bucket_sum) / np.array(bucket_count)

        plt.plot(bucket_x, bucket_max, label=f"{experiment} (max)")
        plt.plot(bucket_x, bucket_min, label=f"{experiment} (min)")
        plt.plot(bucket_x, bucket_mean, label=f"{experiment} (mean)")
    plt.legend()

    mplcursors.cursor(hover=True)
    plt.show()


if __name__ == "__main__":
    main()
