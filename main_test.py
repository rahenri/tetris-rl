#!/usr/bin/env python3

import time
import unittest

import numpy as np
import tensorflow as tf

import main as m


class TestMain(unittest.TestCase):
    shape = [24, 10]

    def test_list_paths(self):
        tf.random.set_seed(0)

        b = m.Board(np.zeros(self.shape, dtype="int8"), "L", 0, 0, 0)
        start = time.time()
        for _i in range(100):
            _visited, finals = b.list_paths()
        print(len(finals))
        print(time.time() - start)
        self.assertEqual(len(finals), 34)

    def test_model(self):
        ones = np.ones(self.shape, dtype=np.float32)
        model = m.AgentModel(self.shape)
        output = model(np.array([ones] * 10, dtype=np.float32))
        np.testing.assert_almost_equal(output, [[0.01319502]] * 10)

        tf.random.set_seed(0)
        output = model(np.array([ones] * 10, dtype=np.float32), training=True).numpy()
        want = [
            [0.06435677],
            [0.05043453],
            [0.0363599],
            [0.02428276],
            [0.02557921],
            [0.03920139],
            [0.02418895],
            [0.01254296],
            [0.03958973],
            [0.0153723],
        ]
        np.testing.assert_almost_equal(output, want)


if __name__ == "__main__":
    unittest.main()
