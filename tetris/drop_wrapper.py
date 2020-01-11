from tetris_engine import Moves

ALLOWED_ACTIONS = {
    Moves.MOVE_LEFT,
    Moves.MOVE_RIGHT,
    Moves.ROTATE_LEFT,
    Moves.ROTATE_RIGHT,
    Moves.DOWN,
}


class DropWrapper:
    def __init__(self, env):
        self.env = env
        self.action_space = env.action_space
        self.observation_space = env.observation_space

    def reset(self):
        return self.env.reset()

    def step(self, actions):
        for action in actions:
            if action not in ALLOWED_ACTIONS:
                raise ValueError(f"Unknown action {action}")
            self.env.step(action)
        return self.env.step(Moves.DROP)

    @property
    def pieces(self):
        return self.env.pieces
