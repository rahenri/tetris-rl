#!/usr/bin/env python3

import time
import unittest

import numpy as np
import tensorflow as tf

from board import Board
from model import AgentModel


class TestMain(unittest.TestCase):
    shape = [24, 10]

    def test_list_paths(self):
        tf.random.set_seed(0)

        b = Board(np.zeros(self.shape, dtype="int8"), "L", 0, 0, 0)
        start = time.time()
        for _i in range(100):
            _visited, finals = b.list_paths()
        print(len(finals))
        print(time.time() - start)
        self.assertEqual(len(finals), 34)

    def test_model(self):
        ones = np.ones(self.shape, dtype=np.float32)
        model = AgentModel("", self.shape)
        output = model(np.array([ones] * 10, dtype=np.float32))
        np.testing.assert_almost_equal(output, [[-0.0006732]] * 10)

        tf.random.set_seed(0)
        output = model(np.array([ones] * 10, dtype=np.float32), training=True).numpy()
        want = [[-0.0006732]] * 10
        np.testing.assert_almost_equal(output, want)


if __name__ == "__main__":
    unittest.main()
