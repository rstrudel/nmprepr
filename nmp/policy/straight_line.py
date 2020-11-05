import numpy as np


class StraightLinePolicy:
    def __init__(self, env):
        self.action_space = env.action_space
        self.env = env

    def reset(self):
        pass

    def get_action(self, obs):
        current = self.env.state.q
        goal = self.env.goal_state.q
        action = (goal - current)[: self.action_space.low.shape[0]]
        norm = np.linalg.norm(action)
        norm = max(0.07, norm)
        action /= norm
        return action, {}
