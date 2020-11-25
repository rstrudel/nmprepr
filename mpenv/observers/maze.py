import numpy as np
import matplotlib.pyplot as plt

from gym import spaces
from gym.spaces import Dict

from mpenv.observers.base import BaseObserver


class MazeObserver(BaseObserver):
    def __init__(self, env):
        super().__init__(env)

        self.obstacle_point_dim = 4
        self.max_edges = 64
        self.obstacles_dim = self.max_edges * self.obstacle_point_dim

        # update observation definition to add the obstacles representation
        self.add_observation("obstacles", self.obstacles_dim)

    def reset(self):
        o = self.env.reset()
        edges = []
        geom_objs = self.env.geoms.geom_objs
        for i, obst in enumerate(geom_objs):
            x, y = obst.placement.translation[:2]
            half_side = obst.geometry.halfSide
            w, h = 2 * half_side[:2]
            w = np.max(w - self.env.thickness, 0)
            h = np.max(h - self.env.thickness, 0)
            edges.append([x - w / 2, y - h / 2, x + w / 2, y + h / 2])
        edges = np.array(edges)
        self.edges = np.zeros((self.max_edges, self.obstacle_point_dim))
        self.edges[: edges.shape[0]] = edges
        o = self.observation(o)
        return o

    def represent_obstacles(self, edges, ee_pos):
        edges = edges.copy()

        # local coordinate frame
        edges[:, [0, 2]] -= ee_pos[0]
        edges[:, [1, 3]] -= ee_pos[1]

        # uncomment for visualization
        # p0 = np.hstack((edges[:, :2], np.zeros((edges.shape[0], 1))))
        # p1 = np.hstack((edges[:, 2:], np.zeros((edges.shape[0], 1))))
        # self.env.o3d_viz.show_lines(p0, p1, blocking=False)

        return edges

    def compute_obs(self, state):
        q, oMi, oMg = state.q_oM
        ee_pos = self.env.robot.get_ee(oMg).translation
        edges = self.represent_obstacles(self.edges, ee_pos)
        return {"edges": edges.flatten()}

    def observation(self, obs):
        state = self.env.get_state()
        current_state, goal_state = state["current"], state["goal"]
        obs_wrapper = self.compute_obs(current_state)

        obs["observation"] = np.concatenate((obs["observation"], obs_wrapper["edges"]))

        return obs
