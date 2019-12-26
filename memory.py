import numpy as np

class Memory:
    def __init__(self, max_size):
        self.max_size = max_size
        self.next_entry_idx = 0
        self.used_indices = 0

        self.boards = np.ones([max_size, 24, 10], dtype=np.int8)
        self.next_boards = np.ones([max_size, 24, 10], dtype=np.int8)
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

    def add(self, info, next_info, reward):
        idx = self._next_idx()

        board = info["board"]
        next_board = next_info["board"] if next_info else self.zero_board
        done = 0 if next_info else 1

        self.boards[idx] = board
        self.next_boards[idx] = next_board
        self.rewards[idx] = reward
        self.dones[idx] = done

    def sample(self, size):
        size = min(size, self.used_indices)
        indices = np.random.choice(self.used_indices, size, replace=False)

        return (
            self.boards[indices],
            self.next_boards[indices],
            self.rewards[indices],
            self.dones[indices],
        )

    def size(self):
        return self.used_indices
