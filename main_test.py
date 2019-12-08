#!/usr/bin/env python3

import time

import numpy as np
import main as m


def main():
    b = m.Board(np.zeros([24, 10], dtype="int8"), "L", 0, 0, 0)
    start = time.time()
    for _ in range(100):
        _, finals = b.list_paths()
    print(len(finals))
    print(time.time() - start)


if __name__ == "__main__":
    main()
