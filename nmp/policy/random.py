import numpy as np


class RandomPolicy:
    def __init__(self, env):
        self.action_space = env.action_space

    def reset(self):
        pass

    def get_action(self, obs):
        # the action space is assumed to be normalized in [-1, 1]
        low = np.array(self.action_space.low, ndmin=1)
        dim = low.shape[0]
        action = np.random.normal(size=(dim,))
        action = np.tanh(action)
        # action = np.clip(action, -1, 1)
        return action, {}
