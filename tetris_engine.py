# Modified from Tetromino by lusob luis@sobrecueva.com
# http://lusob.com
# Released under a "Simplified BSD" license

import random
import pygame
import copy

import numpy as np

FPS = 20
BOXSIZE = 20
BOARDWIDTH = 10
BOARDHEIGHT = 24
WINDOWWIDTH = BOXSIZE * BOARDWIDTH
WINDOWHEIGHT = BOXSIZE * BOARDHEIGHT
BLANK = "."

MOVESIDEWAYSFREQ = 0.15
MOVEDOWNFREQ = 0.1

XMARGIN = 0
TOPMARGIN = 0

#               R    G    B
WHITE = (255, 255, 255)
GRAY = (185, 185, 185)
BLACK = (0, 0, 0)
RED = (155, 0, 0)
LIGHTRED = (175, 20, 20)
GREEN = (0, 155, 0)
LIGHTGREEN = (20, 175, 20)
BLUE = (0, 0, 155)
LIGHTBLUE = (20, 20, 175)
YELLOW = (155, 155, 0)
LIGHTYELLOW = (175, 175, 20)

BORDERCOLOR = BLUE
BGCOLOR = BLACK
TEXTCOLOR = WHITE
TEXTSHADOWCOLOR = GRAY
COLORS = (BLUE, GREEN, RED, YELLOW)
LIGHTCOLORS = (LIGHTBLUE, LIGHTGREEN, LIGHTRED, LIGHTYELLOW)
assert len(COLORS) == len(LIGHTCOLORS)  # each color must have light color

TEMPLATEWIDTH = 5
TEMPLATEHEIGHT = 4

S_SHAPE_TEMPLATE = [
    ["..OO.", ".OO..", ".....", "....."],
    ["..O..", "..OO.", "...O.", "....."],
]

Z_SHAPE_TEMPLATE = [
    [".OO..", "..OO.", ".....", "....."],
    ["..O..", ".OO..", ".O...", "....."],
]

I_SHAPE_TEMPLATE = [
    ["..O..", "..O..", "..O..", "..O.."],
    [".....", "OOOO.", ".....", "....."],
]

O_SHAPE_TEMPLATE = [[".OO..", ".OO..", ".....", "....."]]

J_SHAPE_TEMPLATE = [
    [".O...", ".OOO.", ".....", "....."],
    ["..OO.", "..O..", "..O..", "....."],
    [".OOO.", "...O.", ".....", "....."],
    ["..O..", "..O..", ".OO..", "....."],
]

L_SHAPE_TEMPLATE = [
    ["...O.", ".OOO.", ".....", "....."],
    ["..O..", "..O..", "..OO.", "....."],
    [".OOO.", ".O...", ".....", "....."],
    [".OO..", "..O..", "..O..", "....."],
]

T_SHAPE_TEMPLATE = [
    ["..O..", ".OOO.", ".....", "....."],
    ["..O..", "..OO.", "..O..", "....."],
    [".....", ".OOO.", "..O..", "....."],
    ["..O..", ".OO..", "..O..", "....."],
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

FPSCLOCK = None
DISPLAYSURF = None
BASICFONT = None
BIGFONT = None


class GameState:
    def __init__(self):
        global FPSCLOCK, DISPLAYSURF, BASICFONT, BIGFONT
        pygame.init()
        FPSCLOCK = pygame.time.Clock()
        DISPLAYSURF = pygame.display.set_mode((WINDOWWIDTH, WINDOWHEIGHT))
        BASICFONT = pygame.font.Font("freesansbold.ttf", 18)
        BIGFONT = pygame.font.Font("freesansbold.ttf", 100)
        pygame.display.set_caption("Tetromino")

        # DEBUG
        self.total_lines = 0

        # setup variables for the start of the game
        self.board = self.getBlankBoard()
        self.score = 0
        self.lines = 0
        self.height = 0
        self.pieces = 0

        self.fallingPiece = self.getNewPiece()
        self.nextPiece = self.getNewPiece()

        self.frame_step([1, 0, 0, 0, 0, 0])

        pygame.display.update()

    def getObservationDim(self):
        return (BOARDHEIGHT, BOARDWIDTH, 2)

    def reinit(self):
        self.board = self.getBlankBoard()
        self.score = 0
        self.lines = 0
        self.height = 0
        self.pieces = 0

        self.fallingPiece = self.getNewPiece()
        self.nextPiece = self.getNewPiece()

        self.frame_step([1, 0, 0, 0, 0, 0])

        pygame.display.update()

        self.fallingPiece = self.getNewPiece()
        self.nextPiece = self.getNewPiece()

        self.frame_step([1, 0, 0, 0, 0, 0])

        pygame.display.update()

    def frame_step(self, input):
        did_move = False
        put = False

        # Move left
        if (input[1] == 1) and self.isValidPosition(adjX=-1):
            self.fallingPiece["x"] -= 1
            did_move = True

        # Move right
        elif (input[3] == 1) and self.isValidPosition(adjX=1):
            self.fallingPiece["x"] += 1
            did_move = True

        # Rotating right
        elif input[2] == 1:
            self.fallingPiece["rotation"] = (self.fallingPiece["rotation"] + 1) % len(
                PIECES[self.fallingPiece["shape"]]
            )
            if not self.isValidPosition():
                self.fallingPiece["rotation"] = (
                    self.fallingPiece["rotation"] - 1
                ) % len(PIECES[self.fallingPiece["shape"]])
            else:
                did_move = True

        # Rotating left
        elif input[5] == 1:  # rotate the other direction
            self.fallingPiece["rotation"] = (self.fallingPiece["rotation"] - 1) % len(
                PIECES[self.fallingPiece["shape"]]
            )
            if not self.isValidPosition():
                self.fallingPiece["rotation"] = (
                    self.fallingPiece["rotation"] + 1
                ) % len(PIECES[self.fallingPiece["shape"]])
            else:
                did_move = True

        # Drop Piece the current piece all the way down
        elif input[4] == 1:
            k = 0
            for i in range(1, BOARDHEIGHT):
                if not self.isValidPosition(adjY=i):
                    k = i
                    break
            self.fallingPiece["y"] += k - 1
            did_move = True
            put = True

        if not did_move:
            if not self.isValidPosition(adjY=1):
                put = True
            else:
                self.fallingPiece["y"] += 1

        # let the piece fall if it is time to fall
        # see if the piece has landed
        cleared = 0
        reward = 0
        if put:
            # falling piece has landed, set it on the self.board
            self.addToBoard()

            cleared = self.removeCompleteLines()
            reward = 1 + cleared * 10

            self.lines += cleared
            self.total_lines += cleared

            self.height = self.getHeight()

            self.pieces += 1

            self.fallingPiece = self.nextPiece
            self.nextPiece = self.getNewPiece()

            if not self.isValidPosition():
                image_data = self.simpleState()

                # can't fit a new piece on the self.board, so game over
                return image_data, reward, True, self.info()

        # image_data = pygame.surfarray.array3d(pygame.display.get_surface())
        image_data = self.simpleState()
        return image_data, reward, False, self.info()

    def _render(self, board, falling_piece):
        DISPLAYSURF.fill(BGCOLOR)
        self.drawBoard(board)
        if falling_piece:
            self.drawPiece(falling_piece)
        pygame.display.update()
        image_data = pygame.surfarray.array3d(
            pygame.transform.rotate(pygame.display.get_surface(), 90)
        )
        return image_data

    def getImage(self):
        return self._render(self.board, self.fallingPiece)

    def get_info_image(self, info):
        DISPLAYSURF.fill(BGCOLOR)
        piece = None
        if "piece_shape" in info:
            piece = {
                "shape": info["piece_shape"],
                "x": info["piece_x"],
                "y": info["piece_y"],
                "rotation": info["piece_rotation"],
                "color": info["piece_color"],
            }
        return self._render(info["color_board"], piece)

    def getActionSet(self):
        return list(range(6))

    def getHeight(self):
        stack_height = 0
        for i in range(0, BOARDHEIGHT):
            blank_row = True
            for j in range(0, BOARDWIDTH):
                if self.board[j][i] != ".":
                    blank_row = False
            if not blank_row:
                stack_height = BOARDHEIGHT - i
                break
        return stack_height

    def getNewPiece(self):
        # return a random new piece in a random rotation and color
        shape = random.choice(list(PIECES.keys()))
        newPiece = {
            "shape": shape,
            "rotation": random.randint(0, len(PIECES[shape]) - 1),
            "x": int(BOARDWIDTH / 2) - int(TEMPLATEWIDTH / 2),
            "y": 0,  # start it above the self.board (i.e. less than 0)
            "color": random.randint(0, len(COLORS) - 1),
        }
        return newPiece

    def addToBoard(self):
        # fill in the self.board based on piece's location, shape, and rotation
        for x in range(TEMPLATEWIDTH):
            for y in range(TEMPLATEHEIGHT):
                if (
                    PIECES[self.fallingPiece["shape"]][self.fallingPiece["rotation"]][
                        y
                    ][x]
                    != BLANK
                ):
                    self.board[x + self.fallingPiece["x"]][
                        y + self.fallingPiece["y"]
                    ] = self.fallingPiece["color"]

    def getBlankBoard(self):
        # create and return a new blank self.board data structure
        self.board = []
        for i in range(BOARDWIDTH):
            self.board.append([BLANK] * BOARDHEIGHT)
        return self.board

    def isOnBoard(self, x, y):
        return x >= 0 and x < BOARDWIDTH and y < BOARDHEIGHT

    def isValidPosition(self, adjX=0, adjY=0):
        # Return True if the piece is within the self.board and not colliding
        for x in range(TEMPLATEWIDTH):
            for y in range(TEMPLATEHEIGHT):
                isAboveBoard = y + self.fallingPiece["y"] + adjY < 0
                if (
                    isAboveBoard
                    or PIECES[self.fallingPiece["shape"]][
                        self.fallingPiece["rotation"]
                    ][y][x]
                    == BLANK
                ):
                    continue
                if not self.isOnBoard(
                    x + self.fallingPiece["x"] + adjX, y + self.fallingPiece["y"] + adjY
                ):
                    return False
                if (
                    self.board[x + self.fallingPiece["x"] + adjX][
                        y + self.fallingPiece["y"] + adjY
                    ]
                    != BLANK
                ):
                    return False
        return True

    def isCompleteLine(self, y):
        # Return True if the line filled with boxes with no gaps.
        for x in range(BOARDWIDTH):
            if self.board[x][y] == BLANK:
                return False
        return True

    def removeCompleteLines(self):
        # Remove any completed lines on the self.board, move everything above them down, and return the number of complete lines.
        numLinesRemoved = 0
        y = BOARDHEIGHT - 1  # start y at the bottom of the self.board
        while y >= 0:
            if self.isCompleteLine(y):
                # Remove the line and pull boxes down by one line.
                for pullDownY in range(y, 0, -1):
                    for x in range(BOARDWIDTH):
                        self.board[x][pullDownY] = self.board[x][pullDownY - 1]
                # Set very top line to blank.
                for x in range(BOARDWIDTH):
                    self.board[x][0] = BLANK
                numLinesRemoved += 1
                # Note on the next iteration of the loop, y is the same.
                # This is so that if the line that was pulled down is also
                # complete, it will be removed.
            else:
                y -= 1  # move on to check next row up
        return numLinesRemoved

    def convertToPixelCoords(self, boxx, boxy):
        # Convert the given xy coordinates of the self.board to xy
        # coordinates of the location on the screen.
        return (XMARGIN + (boxx * BOXSIZE)), (TOPMARGIN + (boxy * BOXSIZE))

    def drawBox(self, boxx, boxy, color, pixelx=None, pixely=None):
        # draw a single box (each tetromino piece has four boxes)
        # at xy coordinates on the self.board. Or, if pixelx & pixely
        # are specified, draw to the pixel coordinates stored in
        # pixelx & pixely (this is used for the "Next" piece).
        if color == BLANK:
            return
        if pixelx == None and pixely == None:
            pixelx, pixely = self.convertToPixelCoords(boxx, boxy)
        pygame.draw.rect(
            DISPLAYSURF,
            COLORS[color],
            (pixelx + 1, pixely + 1, BOXSIZE - 1, BOXSIZE - 1),
        )
        pygame.draw.rect(
            DISPLAYSURF,
            LIGHTCOLORS[color],
            (pixelx + 1, pixely + 1, BOXSIZE - 4, BOXSIZE - 4),
        )

    def drawBoard(self, board):
        # draw the border around the self.board
        pygame.draw.rect(
            DISPLAYSURF,
            BORDERCOLOR,
            (
                XMARGIN - 3,
                TOPMARGIN - 7,
                (BOARDWIDTH * BOXSIZE) + 8,
                (BOARDHEIGHT * BOXSIZE) + 8,
            ),
            5,
        )

        # fill the background of the self.board
        pygame.draw.rect(
            DISPLAYSURF,
            BGCOLOR,
            (XMARGIN, TOPMARGIN, BOXSIZE * BOARDWIDTH, BOXSIZE * BOARDHEIGHT),
        )
        # draw the individual boxes on the self.board
        for x in range(BOARDWIDTH):
            for y in range(BOARDHEIGHT):
                self.drawBox(x, y, board[x][y])

    def info(self):
        info = {}

        piece = self.fallingPiece
        rows = []
        for y in range(BOARDHEIGHT):
            row = []
            for x in range(BOARDWIDTH):
                v = self.board[x][y]
                n = 1 if v != BLANK else 0
                row.append(n)
            rows.append(row)
        info["board"] = np.array(rows, dtype="int8")

        info["color_board"] = copy.deepcopy(self.board)

        piece = self.fallingPiece
        if piece is not None:
            info["piece_x"] = piece["x"]
            info["piece_y"] = piece["y"]
            info["piece_shape"] = piece["shape"]
            info["piece_rotation"] = piece["rotation"]
            info["piece_color"] = piece["color"]

        return info

    def simpleState(self):
        piece = self.fallingPiece
        rows = []
        for y in range(BOARDHEIGHT):
            row = []
            for x in range(BOARDWIDTH):
                v = self.board[x][y]
                if v == BLANK:
                    n = 0
                else:
                    n = 1
                row.append([n, 0])
            rows.append(row)

        if piece is not None:
            shapeToDraw = PIECES[piece["shape"]][piece["rotation"]]
            piecex, piecey = piece["x"], piece["y"]

            # draw each of the boxes that make up the piece
            for x in range(TEMPLATEWIDTH):
                for y in range(TEMPLATEHEIGHT):
                    if shapeToDraw[y][x] != BLANK:
                        rows[y + piecey][x + piecex][1] = 1

        return np.array(rows, dtype="int8")

    def drawPiece(self, piece, pixelx=None, pixely=None):
        shapeToDraw = PIECES[piece["shape"]][piece["rotation"]]
        if pixelx == None and pixely == None:
            # if pixelx & pixely hasn't been specified, use the location stored in the piece data structure
            pixelx, pixely = self.convertToPixelCoords(piece["x"], piece["y"])

        # draw each of the boxes that make up the piece
        for x in range(TEMPLATEWIDTH):
            for y in range(TEMPLATEHEIGHT):
                if shapeToDraw[y][x] != BLANK:
                    self.drawBox(
                        None,
                        None,
                        piece["color"],
                        pixelx + (x * BOXSIZE),
                        pixely + (y * BOXSIZE),
                    )

    def drawNextPiece(self):
        # draw the "next" text
        nextSurf = BASICFONT.render("Next:", True, TEXTCOLOR)
        nextRect = nextSurf.get_rect()
        nextRect.topleft = (WINDOWWIDTH - 120, 80)
        DISPLAYSURF.blit(nextSurf, nextRect)
        # draw the "next" piece
        self.drawPiece(self.nextPiece, pixelx=WINDOWWIDTH - 120, pixely=100)
