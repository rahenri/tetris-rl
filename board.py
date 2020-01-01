#!/usr/bin/env python3

import numpy as np
import utils
from numba import jit, int32, int8
from tetris_engine import Moves

S_SHAPE_TEMPLATE = [
    ["..OO.", ".OO..", ".....", ".....", "....."],
    ["..O..", "..OO.", "...O.", ".....", "....."],
]

Z_SHAPE_TEMPLATE = [
    [".OO..", "..OO.", ".....", ".....", "....."],
    ["..O..", ".OO..", ".O...", ".....", "....."],
]

I_SHAPE_TEMPLATE = [
    ["..O..", "..O..", "..O..", "..O..", "....."],
    ["OOOO.", ".....", ".....", ".....", "....."],
]

O_SHAPE_TEMPLATE = [[".OO..", ".OO..", ".....", ".....", "....."]]

J_SHAPE_TEMPLATE = [
    [".O...", ".OOO.", ".....", ".....", "....."],
    ["..OO.", "..O..", "..O..", ".....", "....."],
    [".OOO.", "...O.", ".....", ".....", "....."],
    ["..O..", "..O..", ".OO..", ".....", "....."],
]

L_SHAPE_TEMPLATE = [
    ["...O.", ".OOO.", ".....", ".....", "....."],
    ["..O..", "..O..", "..OO.", ".....", "....."],
    [".OOO.", ".O...", ".....", ".....", "....."],
    [".OO..", "..O..", "..O..", ".....", "....."],
]

T_SHAPE_TEMPLATE = [
    ["..O..", ".OOO.", ".....", ".....", "....."],
    ["..O..", "..OO.", "..O..", ".....", "....."],
    [".OOO.", "..O..", ".....", ".....", "....."],
    ["..O..", ".OO..", "..O..", ".....", "....."],
]

PIECES = {
    "S": S_SHAPE_TEMPLATE,
    "Z": Z_SHAPE_TEMPLATE,
    "J": J_SHAPE_TEMPLATE,
    "L": L_SHAPE_TEMPLATE,
    "I": I_SHAPE_TEMPLATE,
    "O": O_SHAPE_TEMPLATE,
    "T": T_SHAPE_TEMPLATE,
}


ACTIONS = [
    Moves.DOWN,
    Moves.DROP,
    Moves.MOVE_LEFT,
    Moves.MOVE_RIGHT,
    Moves.ROTATE_RIGHT,
    Moves.ROTATE_LEFT,
]


ACTION_MAP = {
    Moves.DOWN: 0,
    Moves.MOVE_LEFT: 1,
    Moves.ROTATE_RIGHT: 2,
    Moves.MOVE_RIGHT: 3,
    Moves.DROP: 4,
    Moves.ROTATE_LEFT: 5,
}


def to_array(shape):
    rows = []
    for row in shape:
        r = []
        for col in row:
            r.append(1 if col == "O" else 0)
        rows.append(r)
    return np.array(rows, dtype="int8")


def rotations_to_array(rotations):
    out = []
    for r in rotations:
        out.append(to_array(r))
    return np.array(out, dtype="int8")


ARRAY_PIECES = {k: rotations_to_array(s) for k, s in PIECES.items()}


@jit(nopython=True)
def _is_valid(board, shape, x, y, rotation):
    return _is_valid_impl(board, shape[rotation], x, y)


@jit(int32(int8[:, :], int8[:, :], int32, int32), nopython=True)
def _is_valid_impl(board, shape, x, y):
    rows = len(board)
    cols = len(board[0])
    for i in range(5):
        for j in range(5):
            if not shape[i][j]:
                continue
            y1 = y + i
            x1 = x + j
            if x1 < 0 or y1 >= rows or x1 >= cols or board[y1][x1]:
                return False
    return True


@jit(nopython=True)
def _drop_piece(board, shape, x, y, rotation):
    return _drop_piece_impl(board, shape[rotation], x, y)


@jit(int32(int8[:, :], int8[:, :], int32, int32), nopython=True)
def _drop_piece_impl(board, shape, x, y):
    while _is_valid_impl(board, shape, x, y + 1):
        y += 1
    return y


@jit(nopython=True)
def _materialize(board, shape, x, y, rotation):
    template = shape[rotation]
    board = np.copy(board)
    rows = len(board)
    cols = len(board[0])
    for i in range(5):
        for j in range(5):
            if not template[i][j]:
                continue
            y1 = y + i
            x1 = x + j
            if x1 < 0 or y1 < 0 or y1 >= rows or x1 >= cols:
                continue
            board[y1][x1] = 1
    unclear = np.zeros((rows, cols), dtype=np.int8)
    j = len(board) - 1
    cleared = 0
    for i in range(len(board) - 1, -1, -1):
        row = board[i]
        if row.sum() < len(row):
            unclear[j] = row
            j -= 1
        else:
            cleared += 1

    return unclear, cleared


@jit
def _action(board, shape, x, y, rotation, action):
    if action == 0:
        # Down
        y += 1
    elif action == 1:
        # Left
        x -= 1
    elif action == 3:
        # Right
        x += 1
    elif action == 2:
        # Rotate right
        rotation = (rotation + 1) % len(shape)
    elif action == 5:
        # Rotate left
        rotation = (rotation - 1) % len(shape)
    elif action == 4:
        # Drop
        y = _drop_piece(board, shape, x, y, rotation)
        b, cleared = _materialize(board, shape, x, y, rotation)
        return (b, x, y, rotation, True, cleared)

    if not _is_valid(board, shape, x, y, rotation):
        return None

    return (board, x, y, rotation, False, 0)


class Board:
    def __init__(self, board, shape, x, y, rotation, done=False, cleared=0):
        self.board = board
        self.shape = shape
        self.x = x
        self.y = y
        self.rotation = rotation
        self.done = done
        self.cleared = cleared

    def action(self, action):
        shape = ARRAY_PIECES[self.shape]
        out = _action(
            self.board, shape, self.x, self.y, self.rotation, ACTION_MAP[action]
        )
        if out is None:
            return out
        board, x, y, rotation, done, cleared = out
        return Board(board, self.shape, x, y, rotation, done, cleared)

    def list_paths(self):
        start = self

        queue = []
        visited = {}
        queue.append(start)
        visited[start.tup()] = True
        finals = []
        while queue:
            node = queue.pop(0)
            for action in ACTIONS:
                next_node = node.action(action)
                if next_node is None:
                    continue
                tup = next_node.tup()
                if tup in visited:
                    continue
                visited[tup] = (action, node)
                if next_node.done:
                    finals.append(next_node)
                else:
                    queue.append(next_node)
        return visited, finals

    def is_done(self):
        return self.done

    def reward(self):
        if not self.is_done():
            return 0
        if self.cleared == 0:
            reward = 0
        elif self.cleared == 1:
            reward = 10
        elif self.cleared == 2:
            reward = 30
        elif self.cleared == 3:
            reward = 60
        elif self.cleared == 4:
            reward = 100
        else:
            raise RuntimeError()
        return reward + 1

    def tup(self):
        return (self.x, self.y, self.rotation, self.done)

    def __repr__(self):
        return str((self.x, self.y, self.rotation, self.done))

    def full_hash(self):
        return utils.hash_value(
            (self.board, self.shape, self.x, self.y, self.rotation, self.done)
        )
