import numpy as np


class SingleObservation:
    def __init__(self, board, turn):
        self.board = board
        self.turn = turn


class ObservationVector:
    def __init__(self, board_shape, size=None):
        self.board_shape = board_shape
        if size is not None:
            self.boards = np.ones([size] + list(board_shape), dtype=np.int8)
            self.turns = np.ones([size], dtype=np.int8)

    def set(self, idx, obs):
        self.boards[idx] = obs.board
        self.turns[idx] = obs.turn

    def select(self, indices):
        out = ObservationVector(self.board_shape)
        out.boards = self.boards[indices]
        out.turns = self.turns[indices]
        return out

    @classmethod
    def from_list(cls, board_shape, obs_list):
        out = ObservationVector(board_shape, len(obs_list))
        for i, obs in enumerate(obs_list):
            out.set(i, obs)
        return out
