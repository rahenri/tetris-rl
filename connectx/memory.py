import numpy as np


class Memory:
    def __init__(self, max_size):
        self.max_size = max_size
        self.next_entry_idx = 0
        self.used_indices = 0

        self.boards = np.ones([max_size, 6, 7], dtype=np.int8)
        self.next_boards = np.ones([max_size, 6, 7], dtype=np.int8)
        self.rewards = np.ones([max_size], dtype=np.int16)
        self.dones = np.ones([max_size], dtype=np.int8)

        self.zero_board = np.ones([24, 10], dtype=np.int8)

    def _next_idx(self):
        idx = self.next_entry_idx
        self.next_entry_idx += 1
        self.used_indices = max(self.used_indices, self.next_entry_idx)
        if self.next_entry_idx >= self.max_size:
            self.next_entry_idx = 0
        return idx

    def add(self, board, next_board, reward):
        idx = self._next_idx()

        done = 1 if next_board is None else 0

        self.boards[idx] = board
        self.rewards[idx] = reward
        self.dones[idx] = done
        if next_board is not None:
            self.next_boards[idx] = next_board

    def sample(self, size):
        indices = np.random.choice(self.used_indices, size, replace=True)

        return (
            self.boards[indices],
            self.next_boards[indices],
            self.rewards[indices],
            self.dones[indices],
        )

    def size(self):
        return self.used_indices
