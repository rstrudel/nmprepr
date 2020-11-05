import numpy as np

import gym
from gym import spaces
from gym.spaces import Dict
import matplotlib.pyplot as plt

from mpenv.observers.base import BaseObserver


class ImageObserver(BaseObserver):
    def __init__(self, env, img_shape, pov, visibility_distance):
        super().__init__(env)

        self.img_shape = img_shape
        self.obstacles_dim = np.prod(self.img_shape)

        self.pov = pov
        if pov not in ["local", "global"]:
            raise ValueError(f"Invalid point of view: {pov}")

        self.visibility_distance = visibility_distance

        # update observation definition to add the obstacles representation
        self.add_observation("obstacles", self.obstacles_dim)

    def reset(self, **kwargs):
        o = self.env.reset(**kwargs)
        self.occ_grid, self.occ_grid_samples = self.env.compute_occupancy_grid(
            self.img_shape[0]
        )
        o = self.observation(o)
        return o

    def compute_obs(self, state, goal_state):
        q, oMi, oMg = state.q_oM
        goal_q, goal_oMi, goal_oMg = goal_state.q_oM

        ee_pos = self.env.robot.get_ee(oMg).translation[:2]
        goal_ee_pos = self.env.robot.get_ee(goal_oMg).translation[:2]

        occ_grid = self.occ_grid.copy()

        nearest_to_robot = np.linalg.norm(
            self.occ_grid_samples - ee_pos, axis=1
        ).argmin()
        nearest_to_goal = np.linalg.norm(
            self.occ_grid_samples - goal_ee_pos, axis=1
        ).argmin()
        occ_grid[nearest_to_robot] = 0.5

        if self.pov == "local":
            squared_occ = occ_grid.reshape(self.img_shape)
            local_occ_grid = np.ones((self.img_shape[0], self.img_shape[1]))
            current_x, current_y = q[:2]
            y_center_pixel, x_center_pixel = (
                int(self.img_shape[0] / 2),
                int(self.img_shape[1] / 2),
            )
            current_x_pixel, current_y_pixel = (
                int(current_x * self.img_shape[1]),
                int(current_y * self.img_shape[0]),
            )
            x_diff = np.abs(x_center_pixel - current_x_pixel)
            y_diff = np.abs(y_center_pixel - current_y_pixel)
            x_offset = x_diff + 1
            y_offset = y_diff + 1

            if (
                current_x_pixel <= self.img_shape[1] / 2
                and current_y_pixel <= self.img_shape[0] / 2
            ):
                local_occ_grid[y_offset:, x_offset:] = squared_occ[
                    :-y_offset, :-x_offset
                ]
            elif (
                current_x_pixel <= self.img_shape[1] / 2
                and current_y_pixel > self.img_shape[0] / 2
            ):
                local_occ_grid[:-y_offset, x_offset:] = squared_occ[
                    y_offset:, :-x_offset
                ]
            elif (
                current_x_pixel > self.img_shape[1] / 2
                and current_y_pixel <= self.img_shape[0] / 2
            ):
                local_occ_grid[y_offset:, :-x_offset] = squared_occ[
                    :-y_offset:, x_offset:
                ]
            else:
                local_occ_grid[:-y_offset, :-x_offset] = squared_occ[
                    y_offset:, x_offset:
                ]

            occ_grid = local_occ_grid.flatten()

        obstacles_img = occ_grid.reshape(self.img_shape)

        # Visualize image
        # plt.ion()
        # plt.imshow(obstacles_img)
        # plt.draw()
        # plt.pause(0.001)
        # plt.ioff()

        obstacles_img = obstacles_img.flatten()
        obstacles_img = (obstacles_img - 0.5) / 0.5

        return {"img": obstacles_img}

    def observation(self, obs):
        state = self.env.get_state()
        current_state, goal_state = state["current"], state["goal"]
        obs_wrapper = self.compute_obs(current_state, goal_state)

        obs["observation"] = np.concatenate((obs["observation"], obs_wrapper["img"]))

        return obs

    def show_representation(self, occ_grid=None):
        if occ_grid is None:
            occ_grid = self.occ_grid
        fig, ax = self.env.render_local_matplotlib()
        colors = np.zeros((np.prod(self.img_shape), 3))
        colors[np.where(occ_grid == 1)[0]] = np.array([1, 0, 0])
        colors[np.where(occ_grid == 0.5)[0]] = np.array([0, 0, 1])
        # colors[np.where(occ_grid == 0.75)[0]] = np.array([0, 1, 0])
        # colors[occ_grid] = np.array([0, 1, 0])
        # for i in range(0, np.prod(self.img_shape)):  # plot occupancy grid
        #     x, y = self.occ_grid_samples[i]
        #     if self.occ_grid[i] == 0:
        #         plt.scatter(x, y, color="red", s=50, alpha=0.7)
        #     else:
        #         plt.scatter(x, y, color="green", s=50, alpha=0.7)
        plt.scatter(
            self.occ_grid_samples[:, 0],
            self.occ_grid_samples[:, 1],
            c=colors,
            s=40,
            alpha=0.8,
        )

        return fig, ax
