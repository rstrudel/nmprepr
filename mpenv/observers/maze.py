import numpy as np
import matplotlib.pyplot as plt

from gym import spaces
from gym.spaces import Dict

from mpenv.observers.base import BaseObserver


class MazeObserver(BaseObserver):
    def __init__(self, env):
        super().__init__(env)

        self.obstacle_point_dim = 4
        # self.visible_cells = 2
        # receptive_field = 2 * self.visible_cells
        receptive_field = self.env.grid_size
        self.max_edges = 2 * receptive_field * (receptive_field + 1)
        # number of edges at last index
        self.obstacles_dim = self.max_edges * self.obstacle_point_dim + 1

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
        self.edges = edges
        o = self.observation(o)
        return o

    def represent_obstacles(self, edges, ee_pos):
        edges = edges.copy()

        # local coordinate frame
        centers = (edges[:, :2] + edges[:, 2:]) / 2
        dist = np.linalg.norm(centers - ee_pos[:2], axis=1)
        edges[:, [0, 2]] -= ee_pos[0]
        edges[:, [1, 3]] -= ee_pos[1]
        # edges = edges[dist < self.visible_cells / self.env.maze.nx]

        # uncomment for visualization
        # p0 = np.hstack((edges[:, :2], np.zeros((edges.shape[0], 1))))
        # p1 = np.hstack((edges[:, 2:], np.zeros((edges.shape[0], 1))))
        # self.env.o3d_viz.show_lines(p0, p1, blocking=False)

        edges_pad = np.zeros((self.max_edges, self.obstacle_point_dim))
        edges_pad[: edges.shape[0]] = edges
        edges_pad = edges_pad.flatten()
        # add number of edges as last index
        edges_pad = np.hstack((edges_pad, edges.shape[0]))

        return edges_pad

    def compute_obs(self, state):
        q, oMi, oMg = state.q_oM
        ee_pos = self.env.robot.get_ee(oMg).translation
        edges = self.represent_obstacles(self.edges, ee_pos)

        return {"edges": edges}

    def observation(self, obs):
        state = self.env.get_state()
        current_state, goal_state = state["current"], state["goal"]
        obs_wrapper = self.compute_obs(current_state)

        obs["observation"] = np.concatenate((obs["observation"], obs_wrapper["edges"]))

        return obs
