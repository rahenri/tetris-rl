import random


class RandomAgent:
    def __init__(self):
        pass

    def act(self, board):
        moves = board.list_moves()
        return random.choice(moves)
