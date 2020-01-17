import random
from board import Position


class RandomAgent:
    def __init__(self):
        pass

    def act(self, board):
        moves = board.list_moves()
        return random.choice(moves)


class GreedyAgent:
    def __init__(self):
        pass

    def act(self, board):
        actions = board.list_moves()
        good_actions = []
        for action in actions:
            board_copy = board.copy()
            ended = board_copy.step(action)
            if ended:
                good_actions.append(action)

        if len(good_actions) > 0:
            return random.choice(good_actions)

        return random.choice(actions)


class BetterGreedyAgent:
    def __init__(self):
        pass

    def act(self, board):
        player = board.turn()
        actions = board.list_moves()
        scored_actions = []
        for action in actions:
            row = board.row(action)
            pos = Position(row, action)

            score = 0
            if board.would_win(pos, player):
                score += 1000000
            if board.would_win(pos, 3 - player):
                score += 1000

            scored_actions.append((action, score))

        scored_actions.sort(key=lambda value: -value[1])
        best_score = scored_actions[0][1]

        best_moves = [action for action, score in scored_actions if score == best_score]

        return random.choice(best_moves)
