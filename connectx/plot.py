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
        "--metric",
        default=["evaluator/reward"],
        type=str,
        help="Metric to plot",
        nargs="*",
    )
    args = parser.parse_args()
    metrics = args.metric
    max_num_buckets = 800
    smoothing = 0.98

    for experiment in args.experiments:
        for metric in metrics:
            filename = os.path.join(args.experiment_dir, experiment, "episodes.txt")
            episode_numbers = []
            values = []
            with open(filename) as episodes:
                for line in episodes:
                    record = json.loads(line)
                    episode_numbers.append(int(record["episode"]))
                    values.append(float(record.get(metric, 0)))

            acc = values[0]
            for i, v in enumerate(values):
                acc = acc * smoothing + v * (1.0 - smoothing)
                values[i] = acc

            minx = np.min(episode_numbers)
            maxx = np.max(episode_numbers)

            num_buckets = min(max_num_buckets, len(values))

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

            # plt.plot(bucket_x, bucket_mean, label=f"{experiment}")
            plt.plot(bucket_x, bucket_mean, label=f"{experiment} {metric}")
    plt.legend()

    mplcursors.cursor(hover=True)
    plt.show()


if __name__ == "__main__":
    main()
