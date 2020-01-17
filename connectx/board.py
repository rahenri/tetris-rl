import copy

import numpy as np


class Position:
    def __init__(self, row: int, col: int):
        self.row = row
        self.col = col

    def add(self, other):
        return Position(self.row + other.row, self.col + other.col)

    def oposite(self):
        return Position(-self.row, -self.col)


DIRECTIONS = [Position(1, 0), Position(0, 1), Position(1, -1), Position(1, 1)]


class Board:
    def __init__(self, height: int, width: int, k: int):
        self.height = height
        self.width = width
        self.k = k
        assert isinstance(self.k, int)
        self._cells = np.zeros([height, width], dtype=np.int8)
        self._next_row = np.zeros(width, dtype=np.int8)
        self._turn = 1
        self._finished = False
        self._winner = None
        self._non_full_cols = width

    def copy(self):
        out = Board(self.height, self.width, self.k)
        out._cells = self._cells.copy()
        out._next_row = self._next_row.copy()
        out._turn = self._turn
        out._finished = self._finished
        out._winner = self._winner
        out._non_full_cols = self._non_full_cols

        return out

    def get(self, pos):
        return self._cells[pos.row, pos.col]

    def _extend(self, player, move, direction):
        count = 0
        for _ in range(self.k - 1):
            move = move.add(direction)
            if (
                move.row < 0
                or move.row >= self.height
                or move.col < 0
                or move.col >= self.width
            ):
                break
            if self.get(move) != player:
                break
            count += 1
        return count

    def row(self, col):
        return self._next_row[col]

    def would_win(self, move, player):
        for direction in DIRECTIONS:
            count = 1 + self._extend(player, move, direction)
            count += self._extend(player, move, direction.oposite())
            if count >= self.k:
                return True
        return False

    def step(self, column):
        assert not self._finished
        assert 0 <= column < self.width
        row = self._next_row[column]
        assert row < self.height
        self._cells[row, column] = self._turn
        self._next_row[column] += 1
        if self._next_row[column] >= self.height:
            self._non_full_cols -= 1

        move = Position(row, column)
        over = self.would_win(move, self._turn)
        if over:
            self._winner = self._turn
            self._finished = True
        elif self._non_full_cols == 0:
            self._finished = True
            self._winner = None

        self._turn = 3 - self._turn

        return self._finished

    def list_moves(self):
        out = []
        for i in range(self.width):
            if self._next_row[i] < self.height:
                out.append(i)
        return out

    def __repr__(self):
        rows = []
        for row in self._cells:
            rows.append("".join(str(c) for c in row))
        rows.reverse()
        return "\n".join(rows)

    def winner(self):
        return self._winner

    def cells(self):
        return self._cells.copy()

    def finished(self):
        return self._finished

    def turn(self):
        return self._turn
