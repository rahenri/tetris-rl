import pygame


BOX_SIZE = 20
BOARD_WIDTH = 10
BOARD_HEIGHT = 24
WINDOW_WIDTH = BOX_SIZE * BOARD_WIDTH
WINDOW_HEIGHT = BOX_SIZE * BOARD_HEIGHT
BLANK = "."

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

COLORS = (BLUE, GREEN, RED, YELLOW)
LIGHTCOLORS = (LIGHTBLUE, LIGHTGREEN, LIGHTRED, LIGHTYELLOW)

LEFT_MARGIN = 0
TOP_MARGIN = 0

BORDER_COLOR = BLUE
BGCOLOR = BLACK
TEXTCOLOR = WHITE
TEXTSHADOWCOLOR = GRAY
COLORS = (BLUE, GREEN, RED, YELLOW)
LIGHTCOLORS = (LIGHTBLUE, LIGHTGREEN, LIGHTRED, LIGHTYELLOW)
assert len(COLORS) == len(LIGHTCOLORS)  # each color must have light color


def _convert_to_pixel_coords(boxx, boxy):
    # Convert the given xy coordinates of the self.board to xy
    # coordinates of the location on the screen.
    return (LEFT_MARGIN + (boxx * BOX_SIZE)), (TOP_MARGIN + (boxy * BOX_SIZE))


class TetrisGUI:
    def __init__(self):
        pygame.init()
        self.fpsclock = pygame.time.Clock()
        self.display_surf = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        self.basic_font = pygame.font.Font("freesansbold.ttf", 18)
        self.big_font = pygame.font.Font("freesansbold.ttf", 100)
        pygame.display.set_caption("Tetromino")

    def _draw_box(self, boxx, boxy, color):
        # draw a single box (each tetromino piece has four boxes)
        # at xy coordinates on the self.board. Or, if pixelx & pixely
        # are specified, draw to the pixel coordinates stored in
        # pixelx & pixely (this is used for the "Next" piece).
        if color == BLANK:
            return
        pixelx, pixely = _convert_to_pixel_coords(boxx, boxy)
        pygame.draw.rect(
            self.display_surf,
            COLORS[color],
            (pixelx + 1, pixely + 1, BOX_SIZE - 1, BOX_SIZE - 1),
        )
        pygame.draw.rect(
            self.display_surf,
            LIGHTCOLORS[color],
            (pixelx + 1, pixely + 1, BOX_SIZE - 4, BOX_SIZE - 4),
        )

    def _draw_board(self, board):
        # draw the border around the self.board
        pygame.draw.rect(
            self.display_surf,
            BORDER_COLOR,
            (
                LEFT_MARGIN - 3,
                TOP_MARGIN - 7,
                (BOARD_WIDTH * BOX_SIZE) + 8,
                (BOARD_HEIGHT * BOX_SIZE) + 8,
            ),
            5,
        )

        # fill the background of the self.board
        pygame.draw.rect(
            self.display_surf,
            BGCOLOR,
            (LEFT_MARGIN, TOP_MARGIN, BOX_SIZE * BOARD_WIDTH, BOX_SIZE * BOARD_HEIGHT),
        )
        # draw the individual boxes on the self.board
        for x in range(BOARD_WIDTH):
            for y in range(BOARD_HEIGHT):
                self._draw_box(x, y, board[x][y])

    def render(self, board):
        self.display_surf.fill(BGCOLOR)
        self._draw_board(board)
        pygame.display.update()

    def screenshot(self):
        image_data = pygame.surfarray.array3d(
            pygame.transform.rotate(self.display_surf, 90)
        )
        return image_data
