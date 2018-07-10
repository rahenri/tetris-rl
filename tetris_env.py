import numpy as np
from gym import spaces
import tetris_engine as game


class TetrisEnv:
    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(self):
        # open up a game state to communicate with emulator
        self.game_state = game.GameState()
        self._action_set = self.game_state.getActionSet()
        self.action_space = spaces.Discrete(len(self._action_set))
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=self.game_state.getObservationDim(),
            dtype=int)
        self._seed = 0

    def step(self, a):
        self._action_set = np.zeros([len(self._action_set)])
        self._action_set[a] = 1
        state, reward, terminal, info = self.game_state.frame_step(
            self._action_set)
        return state, reward, terminal, info

    @property
    def _n_actions(self):
        return len(self._action_set)

    def reset(self):
        return self.game_state.simpleState(), self.game_state.info()

    def render(self):
        return self.game_state.getImage()

    def render_info(self, info):
        return self.game_state.get_info_image(info)
