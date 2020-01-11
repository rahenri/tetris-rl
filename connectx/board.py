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


DIRECTIONS = [Position(1, 0), Position(0, 1), Position(1, -1)]


class Board:
    def __init__(self, height: int, width: int, k: int):
        self.height = height
        self.width = width
        self.k = k
        self._cells = np.zeros([height, width], dtype=np.int8)
        self.next_row = np.zeros(width, dtype=np.int8)
        self.turn = 1
        self._finished = False
        self._winner = None
        self.non_full_cols = width

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

    def has_won(self, move, player):
        for direction in DIRECTIONS:
            count = self._extend(player, move, direction) + self._extend(
                player, move, direction.oposite()
            )
            if count >= self.k:
                return True
        return False

    def act(self, column):
        assert 0 <= column < self.width
        row = self.next_row[column]
        assert row < self.height
        self._cells[row, column] = self.turn
        self.next_row[column] += 1
        if self.next_row[column] >= self.height:
            self.non_full_cols -= 1

        move = Position(row, column)
        over = self.has_won(move, self.turn)
        if over:
            self._winner = self.turn
            self._finished = True

        if self.non_full_cols == 0:
            self._finished = True
            self._winner = None


        self.turn = 3 - self.turn

        return self._finished

    def list_moves(self):
        out = []
        for i in range(self.width):
            if self.next_row[i] < self.height:
                out.append(i)
        return out

    def __repr__(self):
        rows = []
        for row in self._cells:
            rows.append("".join(str(c) for c in row) + "\n")
        rows.reverse()
        return "".join(rows)

    def winner(self):
        return self._winner

    def cells(self):
        return self._cells.copy()

    def cells_if_move(self, col):
        out = self._cells.copy()
        out[self.next_row[col]] = self.turn
        return out

    def finished(self):
        return self._finished
