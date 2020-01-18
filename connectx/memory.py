import numpy as np
from observation import ObservationVector


class Memory:
    def __init__(self, board_shape, max_size):
        self.max_size = max_size
        self.next_entry_idx = 0
        self.used_indices = 0

        board_shape = list(board_shape)

        self.obs = ObservationVector(board_shape, max_size)
        self.next_obs = ObservationVector(board_shape, max_size)
        self.rewards = np.ones([max_size], dtype=np.int16)
        self.dones = np.ones([max_size], dtype=np.int8)

    def _next_idx(self):
        idx = self.next_entry_idx
        self.next_entry_idx += 1
        self.used_indices = max(self.used_indices, self.next_entry_idx)
        if self.next_entry_idx >= self.max_size:
            self.next_entry_idx = 0
        return idx

    def add(self, obs, next_obs, reward):
        idx = self._next_idx()

        done = 1 if next_obs is None else 0

        self.obs.set(idx, obs)
        self.rewards[idx] = reward
        self.dones[idx] = done
        if next_obs is not None:
            self.next_obs.set(idx, next_obs)

    def sample(self, size):
        indices = np.random.choice(self.used_indices, size, replace=True)

        return (
            self.obs.select(indices),
            self.next_obs.select(indices),
            self.rewards[indices],
            self.dones[indices],
        )

    def size(self):
        return self.used_indices
