#!/usr/bin/env python3

import main as m
import numpy as np
import time


def main():
    b = m.Board(np.zeros([24, 10], dtype='int8'), 'L', 0, 0, 0)
    start = time.time()
    for i in range(100):
        visited, finals = b.list_paths()
    print(len(finals))
    print(time.time() - start)

    pass


if __name__ == '__main__':
    main()
