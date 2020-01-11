import tetris_engine as game


class Discrete:
    def __init__(self, n):
        self.n = n


class Box:
    def __init__(self, low, high, shape, dtype):
        self.low = low
        self.high = high
        self.shape = shape
        self.dtype = dtype


class TetrisEnv:
    def __init__(self):
        # open up a game state to communicate with emulator
        self.game_state = game.GameState()
        self._action_set = self.game_state.get_action_set()

        self.action_space = Discrete(len(self._action_set))
        self.observation_space = Box(
            low=0, high=255, shape=self.game_state.get_observation_dim(), dtype=int
        )
        self._seed = 0

    def step(self, action):
        return self.game_state.step(action)

    @property
    def _n_actions(self):
        return len(self._action_set)

    @property
    def pieces(self):
        return self.game_state.pieces

    def reset(self):
        self.game_state.reinit()
        return self.game_state.simple_state(), self.game_state.info()
