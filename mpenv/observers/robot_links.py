import numpy as np
import pinocchio as pin

from gym import spaces
from gym.spaces import Dict

from mpenv.observers.base import BaseObserver
from mpenv.core import utils


class RobotLinksObserver(BaseObserver):
    def __init__(self, env, coordinate_frame):
        super().__init__(env)
        self.coordinate_frame = coordinate_frame
        if coordinate_frame not in ["local", "global"]:
            raise ValueError(f"Invalid coordinate system: {coordinate_frame}")

        self.robot_name = env.robot_name
        self.robot_props = env.robot_props
        self.n_joints = self.robot_props["n_joints"]
        self.link_dim = self.robot_props[coordinate_frame]["link_dim"]

        # robot links (child link of joint) + goal "link"
        self.config_dim = self.n_joints * self.link_dim
        self.robot_props[coordinate_frame]["config_dim"] = self.config_dim
        self.goal_dim = self.robot_props[coordinate_frame]["goal_dim"]
        self.goal_rep_dim = self.robot_props[coordinate_frame]["goal_rep_dim"]

        self.add_observation("config", self.config_dim)
        self.add_observation("goal", self.goal_rep_dim)

        self.update_observation_box("desired_goal", self.goal_dim)
        self.update_observation_box("achieved_goal", self.goal_dim)
        self.update_observation_box("representation_goal", self.goal_rep_dim)

    def represent_goal(self, achieved_goal, desired_goal):
        if self.coordinate_frame == "local":
            achieved_goal = achieved_goal.reshape(-1, self.n_joints, 4, 4)
            desired_goal = desired_goal.reshape(-1, self.n_joints, 4, 4)
            diff = np.linalg.inv(achieved_goal) @ desired_goal
            motions = np.zeros((diff.shape[0], self.n_joints, 6))
            for i, d in enumerate(diff):
                for j, dj in enumerate(d):
                    m = pin.log6(dj)
                    motions[i, j, :3] = m.linear
                    motions[i, j, 3:] = m.angular
            goal_repr = motions
            goal_repr = goal_repr[:, :, : self.goal_rep_dim]
        elif self.coordinate_frame == "global":
            goal_repr = desired_goal[:, : self.goal_rep_dim]

        return goal_repr.reshape(goal_repr.shape[0], -1)

    def compute_obs(self, state, goal_state):
        q, oMi, oMg = state.q_oM
        goal_q, goal_oMi, goal_oMg = goal_state.q_oM

        if self.coordinate_frame == "local":
            achieved_oMg = self.env.robot.get_oMg_np(oMg)
            desired_oMg = self.env.robot.get_oMg_np(goal_oMg)
            goal_repr = self.represent_goal(
                achieved_oMg.flatten()[None], desired_oMg.flatten()[None]
            )[0]
        elif self.coordinate_frame == "global":
            goal_repr = self.represent_goal(q[None], goal_q[None])[0]

        config_repr = self.robot.get_representation(q, oMi, oMg)
        config_repr = config_repr.flatten()

        config_repr = config_repr[: self.config_dim]
        config_repr = config_repr.flatten().copy()
        obs_dict = {
            "config": config_repr,
            "goal_repr": goal_repr,
            "achieved_q": q,
            "desired_q": goal_q,
        }

        if self.coordinate_frame == "local":
            obs_dict["achieved_goal"] = achieved_oMg
            obs_dict["desired_goal"] = desired_oMg
        elif self.coordinate_frame == "global":
            obs_dict["achieved_goal"] = q
            obs_dict["desired_goal"] = goal_q

        return obs_dict

    def observation(self, obs):
        state = self.env.get_state()
        current_state, goal_state = state["current"], state["goal"]
        obs_wrapper = self.compute_obs(current_state, goal_state)

        obs["achieved_goal"] = obs_wrapper["achieved_goal"]
        obs["desired_goal"] = obs_wrapper["desired_goal"]
        obs["achieved_q"] = obs_wrapper["achieved_q"]
        obs["desired_q"] = obs_wrapper["desired_q"]
        obs["representation_goal"] = obs_wrapper["goal_repr"]
        obs["observation"] = np.concatenate((obs["observation"], obs_wrapper["config"]))

        return obs
