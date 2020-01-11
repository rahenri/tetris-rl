#!/usr/bin/env python3

import numpy as np
import psutil

import train
from memory import Memory

import os
import time


def main():
    p = psutil.Process(os.getpid())

    mem_size = 10000000
    memory = Memory(mem_size)
    agent = train.NNAgent((10, 24), 6)
    info = {
        "board": np.zeros((24, 10), dtype=np.int8),
    }
    for _ in range(mem_size):
        memory.add(info, info, 0)
    for i in range(10000000000):
        start = time.time()
        agent.train(memory, 1 << 14)
        end = time.time()
        rss = p.memory_info().rss / 1024 / 1024
        duration = end - start
        print(f"{i}: Memory: {rss:.1f}GB, Duration (sec): {duration:.1f}")


if __name__ == "__main__":
    main()
