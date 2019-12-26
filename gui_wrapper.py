class GuiWrapper:
    def __init__(self, env, gui):
        self.env = env
        self.gui = gui
        self.action_space = env.action_space
        self.observation_space = env.observation_space

    def reset(self):
        return self.env.reset()

    def step(self, action):
        state, reward, done, info = self.env.step(action)
        self.gui.render(info["color_board_with_falling_piece"])
        if not self.gui.process_events():
            raise SystemExit()
        return state, reward, done, info

    @property
    def pieces(self):
        return self.env.pieces
