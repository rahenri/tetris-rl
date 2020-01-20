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
    parser.add_argument(
        "--metric", default="reward", type=str, help="Metric to plot",
    )
    args = parser.parse_args()
    metric = args.metric
    max_num_buckets = 400
    smoothing = 0.9

    for experiment in args.experiments:
        filename = os.path.join(args.experiment_dir, experiment, "episodes.txt")
        episode_numbers = []
        values = []
        with open(filename) as episodes:
            for line in episodes:
                record = json.loads(line)
                episode_numbers.append(int(record["episode"]))
                values.append(float(record.get(metric, 0)))
        minx = np.min(episode_numbers)
        maxx = np.max(episode_numbers)

        num_buckets = min(max_num_buckets, len(values))

        bucket_max = [None] * num_buckets
        bucket_min = [None] * num_buckets
        bucket_sum = [0] * num_buckets
        bucket_count = [0] * num_buckets

        bucket_x = []
        for i in range(num_buckets):
            bucket_x.append(minx + (maxx - minx) * i / num_buckets)

        for i, value in enumerate(values):
            x = min(
                int((episode_numbers[i] - minx) / (maxx - minx) * num_buckets),
                num_buckets - 1,
            )

            bucket_sum[x] += value
            bucket_count[x] += 1

        bucket_mean = np.array(bucket_sum) / np.array(bucket_count)
        bucket_mean_smoothed = bucket_mean.copy()

        last = bucket_mean[0]
        for i in range(1, len(bucket_mean)):
            last = last * smoothing + bucket_mean[i] * (1.0 - smoothing)
            bucket_mean_smoothed[i] = last

        plt.plot(bucket_x, bucket_mean, label=f"{experiment}")
        plt.plot(bucket_x, bucket_mean_smoothed, label=f"{experiment} smoothed")
    plt.legend()

    mplcursors.cursor(hover=True)
    plt.show()


if __name__ == "__main__":
    main()
