# Modified from Tetromino by lusob luis@sobrecueva.com
# http://lusob.com
# Released under a "Simplified BSD" license

import random
import copy

import numpy as np

BOARD_WIDTH = 10
BOARD_HEIGHT = 24
BLANK = "."


TEMPLATE_WIDTH = 5
TEMPLATE_HEIGHT = 5

NUM_COLORS = 4

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


class GameState:
    def __init__(self):
        # DEBUG
        self.total_lines = 0

        # setup variables for the start of the game
        self.board = self.get_blank_board()
        self.score = 0
        self.lines = 0
        self.height = 0
        self.pieces = 0

        self.falling_piece = self.get_new_piece()
        self.next_piece = self.get_new_piece()

        self.frame_step([1, 0, 0, 0, 0, 0])

    def reinit(self):
        self.board = self.get_blank_board()
        self.score = 0
        self.lines = 0
        self.height = 0
        self.pieces = 0

        self.falling_piece = self.get_new_piece()
        self.next_piece = self.get_new_piece()

        self.frame_step([1, 0, 0, 0, 0, 0])

        self.falling_piece = self.get_new_piece()
        self.next_piece = self.get_new_piece()

        self.frame_step([1, 0, 0, 0, 0, 0])

    def frame_step(self, action):
        did_move = False
        put = False

        # Move left
        if (action[1] == 1) and self.is_valid_position(adj_x=-1):
            self.falling_piece["x"] -= 1
            did_move = True

        # Move right
        elif (action[3] == 1) and self.is_valid_position(adj_x=1):
            self.falling_piece["x"] += 1
            did_move = True

        # Rotating right
        elif action[2] == 1:
            self.falling_piece["rotation"] = (self.falling_piece["rotation"] + 1) % len(
                PIECES[self.falling_piece["shape"]]
            )
            if not self.is_valid_position():
                self.falling_piece["rotation"] = (
                    self.falling_piece["rotation"] - 1
                ) % len(PIECES[self.falling_piece["shape"]])
            else:
                did_move = True

        # Rotating left
        elif action[5] == 1:  # rotate the other direction
            self.falling_piece["rotation"] = (self.falling_piece["rotation"] - 1) % len(
                PIECES[self.falling_piece["shape"]]
            )
            if not self.is_valid_position():
                self.falling_piece["rotation"] = (
                    self.falling_piece["rotation"] + 1
                ) % len(PIECES[self.falling_piece["shape"]])
            else:
                did_move = True

        # Drop Piece the current piece all the way down
        elif action[4] == 1:
            k = 0
            for i in range(1, BOARD_HEIGHT):
                if not self.is_valid_position(adj_y=i):
                    k = i
                    break
            self.falling_piece["y"] += k - 1
            did_move = True
            put = True

        if not did_move:
            if not self.is_valid_position(adj_y=1):
                put = True
            else:
                self.falling_piece["y"] += 1

        # let the piece fall if it is time to fall
        # see if the piece has landed
        cleared = 0
        reward = 0
        if put:
            # falling piece has landed, set it on the self.board
            self.add_to_board()

            cleared = self.remove_complete_lines()
            reward = 1 + cleared * 10

            self.lines += cleared
            self.total_lines += cleared

            self.height = self.get_height()

            self.pieces += 1

            self.falling_piece = self.next_piece
            self.next_piece = self.get_new_piece()

            if not self.is_valid_position():
                image_data = self.simple_state()

                # can't fit a new piece on the self.board, so game over
                return image_data, reward, True, self.info()

        image_data = self.simple_state()
        return image_data, reward, False, self.info()

    def get_action_set(self):
        return list(range(6))

    def get_observation_dim(self):
        return (BOARD_HEIGHT, BOARD_WIDTH, 2)

    def get_height(self):
        stack_height = 0
        for i in range(0, BOARD_HEIGHT):
            blank_row = True
            for j in range(0, BOARD_WIDTH):
                if self.board[j][i] != ".":
                    blank_row = False
            if not blank_row:
                stack_height = BOARD_HEIGHT - i
                break
        return stack_height

    def get_new_piece(self):
        # return a random new piece in a random rotation and color
        shape = random.choice(list(PIECES.keys()))
        new_piece = {
            "shape": shape,
            "rotation": random.randint(0, len(PIECES[shape]) - 1),
            "x": int(BOARD_WIDTH / 2) - int(TEMPLATE_WIDTH / 2),
            "y": 0,  # start it above the self.board (i.e. less than 0)
            "color": random.randint(0, NUM_COLORS - 1),
        }
        return new_piece

    def add_to_board(self):
        # fill in the self.board based on piece's location, shape, and rotation
        for x in range(TEMPLATE_WIDTH):
            for y in range(TEMPLATE_HEIGHT):
                if (
                    PIECES[self.falling_piece["shape"]][self.falling_piece["rotation"]][
                        y
                    ][x]
                    != BLANK
                ):
                    self.board[x + self.falling_piece["x"]][
                        y + self.falling_piece["y"]
                    ] = self.falling_piece["color"]

    def get_blank_board(self):
        # create and return a new blank self.board data structure
        self.board = []
        for _ in range(BOARD_WIDTH):
            self.board.append([BLANK] * BOARD_HEIGHT)
        return self.board

    def is_on_board(self, x, y):
        return 0 <= x < BOARD_WIDTH and y < BOARD_HEIGHT

    def is_valid_position(self, adj_x=0, adj_y=0):
        # Return True if the piece is within the self.board and not colliding
        for x in range(TEMPLATE_WIDTH):
            for y in range(TEMPLATE_HEIGHT):
                is_above_board = y + self.falling_piece["y"] + adj_y < 0
                if (
                    is_above_board
                    or PIECES[self.falling_piece["shape"]][
                        self.falling_piece["rotation"]
                    ][y][x]
                    == BLANK
                ):
                    continue
                if not self.is_on_board(
                    x + self.falling_piece["x"] + adj_x,
                    y + self.falling_piece["y"] + adj_y,
                ):
                    return False
                if (
                    self.board[x + self.falling_piece["x"] + adj_x][
                        y + self.falling_piece["y"] + adj_y
                    ]
                    != BLANK
                ):
                    return False
        return True

    def is_complete_line(self, y):
        # Return True if the line filled with boxes with no gaps.
        for x in range(BOARD_WIDTH):
            if self.board[x][y] == BLANK:
                return False
        return True

    def remove_complete_lines(self):
        # Remove any completed lines on the self.board, move everything above them down, and return the number of complete lines.
        num_lines_removed = 0
        y = BOARD_HEIGHT - 1  # start y at the bottom of the self.board
        while y >= 0:
            if self.is_complete_line(y):
                # Remove the line and pull boxes down by one line.
                for pull_down_y in range(y, 0, -1):
                    for x in range(BOARD_WIDTH):
                        self.board[x][pull_down_y] = self.board[x][pull_down_y - 1]
                # Set very top line to blank.
                for x in range(BOARD_WIDTH):
                    self.board[x][0] = BLANK
                num_lines_removed += 1
                # Note on the next iteration of the loop, y is the same.
                # This is so that if the line that was pulled down is also
                # complete, it will be removed.
            else:
                y -= 1  # move on to check next row up
        return num_lines_removed

    def info(self):
        info = {}

        piece = self.falling_piece
        rows = []
        for y in range(BOARD_HEIGHT):
            row = []
            for x in range(BOARD_WIDTH):
                v = self.board[x][y]
                n = 1 if v != BLANK else 0
                row.append(n)
            rows.append(row)
        info["board"] = np.array(rows, dtype="int8")

        info["color_board"] = copy.deepcopy(self.board)

        color_board_with_falling_piece = copy.deepcopy(self.board)
        info["color_board_with_falling_piece"] = color_board_with_falling_piece

        piece = self.falling_piece
        if piece is not None:
            info["piece_x"] = piece["x"]
            info["piece_y"] = piece["y"]
            info["piece_shape"] = piece["shape"]
            info["piece_rotation"] = piece["rotation"]
            info["piece_color"] = piece["color"]

            shape_to_draw = PIECES[piece["shape"]][piece["rotation"]]
            piecex, piecey = piece["x"], piece["y"]

            # draw each of the boxes that make up the piece
            for x in range(TEMPLATE_WIDTH):
                for y in range(TEMPLATE_HEIGHT):
                    y1 = y + piecey
                    x1 = x + piecex
                    if x < 0 or x >= BOARD_WIDTH or y < 0 or y >= BOARD_HEIGHT:
                        continue
                    if shape_to_draw[y][x] != BLANK:
                        color_board_with_falling_piece[x1][y1] = piece["color"]

        return info

    def simple_state(self):
        piece = self.falling_piece
        rows = []
        for y in range(BOARD_HEIGHT):
            row = []
            for x in range(BOARD_WIDTH):
                v = self.board[x][y]
                if v == BLANK:
                    n = 0
                else:
                    n = 1
                row.append([n, 0])
            rows.append(row)

        if piece is not None:
            shape_to_draw = PIECES[piece["shape"]][piece["rotation"]]
            piecex, piecey = piece["x"], piece["y"]

            # draw each of the boxes that make up the piece
            for x in range(TEMPLATE_WIDTH):
                for y in range(TEMPLATE_HEIGHT):
                    if shape_to_draw[y][x] != BLANK:
                        rows[y + piecey][x + piecex][1] = 1

        return np.array(rows, dtype="int8")
