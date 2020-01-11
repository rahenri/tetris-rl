import random

class RandomAgent:
    def __init__(self, player):
        self.player = player

    def act(self, board):
        moves = board.list_moves()
        return random.choice(moves)
