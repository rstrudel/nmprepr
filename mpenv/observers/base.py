import numpy as np

import gym


class BaseObserver(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        # obs_indices store the indices at which data is stored once the vector is flattened
        # useful to recover structured data in a model
        self.obs_shape = self.env.obs_shape
        self.obs_indices = self.env.obs_indices
        self.observation_space = self.env.observation_space

    def set_eval(self):
        self.env.set_eval()

    def observe(self):
        observation = self.env.observe()
        return self.observation(observation)

    def add_observation(self, name, obs_size):
        """
        check obs_indices definition in __init__
        handle rlkit concatenation logic:
        if the goal dimension is passed, it should not count in the global observation shape
        given (o, g) a tuple of (observation, goal), rlkit computes
        a = policy((o, g)), g is thus always put at the end
        """
        if name is "goal":
            self.obs_indices[name] = slice(-obs_size, None)
        else:
            self.obs_indices[name] = slice(self.obs_shape, self.obs_shape + obs_size)
            self.obs_shape += obs_size
        self.update_observation_box("observation", self.obs_shape)

    def update_observation_box(self, name, shape):
        box = self.observation_space[name]
        box.low = -np.ones(shape)
        box.high = np.ones(shape)
        box.shape = (shape,)
