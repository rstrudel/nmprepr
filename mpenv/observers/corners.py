import numpy as np
import matplotlib.pyplot as plt

from gym import spaces
from gym.spaces import Dict

from mpenv.observers.base import BaseObserver


class CornersObserver(BaseObserver):
    def __init__(self, env, coordinate_frame):
        super().__init__(env)

        self.coordinate_frame = coordinate_frame
        self.obstacle_point_dim = 2
        self.n_obstacles = 5
        self.n_corners = 2
        self.obstacles_dim = self.n_obstacles * self.n_corners * self.obstacle_point_dim

        # update observation definition to add the obstacles representation
        self.add_observation("obstacles", self.obstacles_dim)

    def reset(self):
        o = self.env.reset()
        corners = []
        geom_objs = self.env.geoms.geom_objs
        for i, obst in enumerate(geom_objs):
            x, y = obst.placement.translation[:2]
            half_side = obst.geometry.halfSide
            w, h = 2 * half_side[:2]
            corners.append([x - w / 2, x + w / 2, y - h / 2, y + h / 2])
        self.corners = np.array(corners)
        o = self.observation(o)
        return o

    def represent_obstacles(self, corners, ee_pos):
        obstacles_repr = corners.copy()

        if self.coordinate_frame == "local":
            obstacles_repr[:, :2] -= ee_pos[0]
            obstacles_repr[:, 2:] -= ee_pos[1]

        return obstacles_repr

    def compute_obs(self, state):
        q, oMi, oMg = state.q_oM
        ee_pos = self.env.robot.get_ee(oMg).translation
        corners = self.represent_obstacles(self.corners, ee_pos)
        return {"corners": corners.flatten()}

    def observation(self, obs):
        state = self.env.get_state()
        current_state, goal_state = state["current"], state["goal"]
        obs_wrapper = self.compute_obs(current_state)

        obs["observation"] = np.concatenate(
            (obs["observation"], obs_wrapper["corners"])
        )

        return obs

    def show_representation(self):
        ax = self.env.render_matplotlib()
        corners = self.represent_obstacles(self.corners, ee_pos=np.zeros(3))
        corners_x = corners[:, :2].flatten()
        corners_y = corners[:, 2:].flatten()
        ax.scatter(corners_x, corners_y, marker="o", c="red", s=40, alpha=0.8)
        return ax
