import time
import os
import numpy as np
import hppfcl
import pinocchio as pin

import gym
from gym.spaces import Dict
from gym import spaces
from gym.utils import seeding

from mpenv.core.model import ModelWrapper
from mpenv.core.model import ConfigurationWrapper

from mpenv.envs import utils
from mpenv.core.visualizer import Visualizer
from mpenv.core.o3d_visualizer import Open3DVisualizer

from mpenv.core.mesh import Mesh

from mpenv.robot.freeflyer import FreeFlyer


class Base(gym.Env):
    def __init__(self, robot_name):
        self.seed()
        self.delta_collision_check = 1e-2

        self.model = None
        # gepetto, panda3d, meshcat
        self.viz_name = "gepetto"
        self.viz = None
        self.o3d_viz = Open3DVisualizer()
        self.state = None
        self.goal_state = None
        self.cartesian_integration = False
        self._seed = None
        self.config_dim = 0
        self.obstacles_dim = 0
        self.obstacle_point_dim = 0
        self.goal_dim = 0
        self.robot_name = robot_name
        self.model_wrapper = ModelWrapper()

        alpha = 0.3
        self.robot_color = np.array([1, 0.65, 0, alpha])
        self.start_color = np.array([1, 0, 0, alpha])
        self.goal_color = np.array([0, 1, 0, alpha])
        self.n_obstacles = None
        self.obstacles_color = None
        self.obstacles_alpha = 0.8
        self.robot = self.add_robot(self.robot_name, bounds=None)

        self.info_sizes = {"collided": 1}
        self.dict_reward = {"goal": 20, "free": -0.1, "collision": -4}

        # obs_indices store the indices at which data is stored once the vector is flattened
        # useful to recover structured data in a model
        self.obs_shape = 0
        self.obs_indices = {}
        self.observation_space = Dict(
            {
                "observation": spaces.Box(
                    low=-1.0, high=1.0, shape=(0,), dtype=np.float32,
                ),
                "desired_goal": spaces.Box(
                    low=-1.0, high=1.0, shape=(0,), dtype=np.float32
                ),
                "achieved_goal": spaces.Box(
                    low=-1.0, high=1.0, shape=(0,), dtype=np.float32
                ),
                "representation_goal": spaces.Box(
                    low=-1.0, high=1.0, shape=(0,), dtype=np.float32,
                ),
            }
        )

    def add_robot(self, robot_name, bounds):
        model_wrapper = self.model_wrapper
        color = self.robot_color
        if robot_name == "sphere":
            radius = 0.03
            geom = hppfcl.Sphere(radius)
            sphere_mesh = Mesh(name="robot", geometry=geom, color=color)
            robot = FreeFlyer(model_wrapper, sphere_mesh, bounds)
        elif robot_name == "sphere2d":
            radius = 0.01
            geom = hppfcl.Sphere(radius)
            sphere_mesh = Mesh(name="robot", geometry=geom, color=color)
            robot = FreeFlyer(model_wrapper, sphere_mesh, bounds)
        elif robot_name == "s_shape":
            mesh_path = "../assets/s_shape_description/s_shape.stl"
            scale = (0.2, 0.2, 0.2)
            s_mesh = Mesh(
                name="robot", geometry_path=mesh_path, color=color, scale=scale
            )
            robot = FreeFlyer(model_wrapper, s_mesh, bounds)
        else:
            raise ValueError(f"Unknown robot: {robot_name}")
        self.robot_n_joints = robot.n_joints
        return robot

    def add_obstacle(self, geom_obj, static):
        if static:
            geom_model = self.model_wrapper.geom_model
            geom_model.addGeometryObject(geom_obj)
        else:
            raise NotImplementedError

        # check collisions between the robot and obstacles
        robot_n_joints = self.robot_n_joints
        check_collision = range(robot_n_joints)
        n_geom_model = len(geom_model.geometryObjects)
        for collision_id in check_collision:
            geom_model.addCollisionPair(
                pin.CollisionPair(collision_id, n_geom_model - 1)
            )

    def set_state(self, qw):
        if isinstance(qw, np.ndarray):
            qw = ConfigurationWrapper(self.model_wrapper, qw)
        self.state = qw

    def set_goal_state(self, goal_qw):
        if isinstance(goal_qw, np.ndarray):
            goal_qw = ConfigurationWrapper(self.model_wrapper, goal_qw)
        self.goal_state = goal_qw

    def random_configuration(self, only_free=True):
        if only_free:
            state = self.model_wrapper.random_free_configuration()
        else:
            state = self.model_wrapper.random_configuration()
        _, state = self.model_wrapper.clip(
            state, self.robot.bounds[0], self.robot.bounds[1]
        )
        return state

    def seed(self, seed=None):
        if seed is not None:
            pin.seed(seed)
        np_random, seed = seeding.np_random(seed)
        self._np_random = np_random
        self._seed = seed
        return [seed]

    def reset(self, **kwargs):
        self.model_wrapper = ModelWrapper()
        self.viz = None
        self.showed_goal = False
        return self._reset(**kwargs)

    def _reset(self, **kwargs):
        raise NotImplementedError

    def load_dataset(self, dataset_path):
        if not os.path.exists(dataset_path):
            raise ValueError(f"No dataset found: {dataset_path}")
        dataset_geoms = utils.load_dataset_geoms(dataset_path)
        return dataset_geoms

    def stopping_configuration(self, path):
        """
        Assumes path[0] is always a collision free configuration
        Returns the latest configuration of path which is collision free
        """
        model_wrapper = self.model_wrapper
        new_state = path[0]
        collision_labels = np.zeros_like(model_wrapper.collision_labels())
        for state in path[1:]:
            collide = model_wrapper.collision(state)
            collision_labels = model_wrapper.collision_labels()
            if not collide:
                new_state = state
            else:
                return new_state, collision_labels
        return new_state, collision_labels

    def move(self, state, velocity):
        model_wrapper = self.model_wrapper
        # velocity = self.entities.lift_speed(velocity)
        next_state = model_wrapper.integrate(
            state, velocity, self.cartesian_integration
        )
        path = model_wrapper.arange(state, next_state, self.delta_collision_check)
        next_state_free, collision_labels = self.stopping_configuration(path)
        return next_state_free, collision_labels

    def format_action(self, action):
        action = np.clip(action, -1, 1)
        action_range = self.robot_props["action_range"]
        action_scaled = action * action_range
        velocity = np.zeros(6)
        velocity[: self.robot_props["action_dim"]] = action_scaled
        return velocity

    def step(self, action, clip_next_state=True):
        model_wrapper = self.model_wrapper
        action_move = self.format_action(action)
        new_state, collision_labels = self.move(self.state, action_move)
        collided = collision_labels.any(0, keepdims=True)
        if clip_next_state:
            clipped, new_state = model_wrapper.clip(
                new_state, self.robot.bounds[0], self.robot.bounds[1]
            )

        if self.viz:
            previous_oMg = self.state.q_oM[2]
            current_oMg = new_state.q_oM[2]
            previous_ee = self.robot.get_ee(previous_oMg).translation
            current_ee = self.robot.get_ee(current_oMg).translation
            self.viz.add_edge_to_roadmap("path", previous_ee, current_ee)

        self.state = new_state

        q, oMi, oMg = self.state.q_oM
        goal_q, goal_oMi, goal_oMg = self.goal_state.q_oM

        achieved_oMg = self.robot.get_oMg_np(oMg).flatten()
        desired_oMg = self.robot.get_oMg_np(goal_oMg).flatten()
        reward, done, success = self.compute_rewards(
            achieved_oMg[None, :],
            desired_oMg[None, :],
            action[None, :],
            collided[None, :],
        )
        reward, done, success = reward[0], done[0], success[0]
        self.done = done

        info = {"collided": collided, "success": success}

        return self.observation(), reward, done, info

    def compute_rewards(self, achieved_goal, goal, action, collided):
        """
        if ||achieved_goal-goal|| < dist_goal, return -d(a_goal, goal) only if not in collision
        if ||achieved_goal-goal|| > dist_goal, return free or collision reward
        else return previous reward
        """
        dist_goal_success = self.robot_props["dist_goal"]
        collided = collided.astype(bool)
        n_joints = self.robot_n_joints
        # links defined goal
        achieved_goal = achieved_goal.reshape(-1, n_joints, 4, 4)
        goal = goal.reshape(-1, n_joints, 4, 4)
        diff = np.linalg.inv(achieved_goal) @ goal
        motions = np.zeros((diff.shape[0], n_joints, 6))
        for i, d in enumerate(diff):
            for j, dj, in enumerate(d):
                m = pin.log6(dj)
                motions[i, j, :3] = m.linear
                motions[i, j, 3:] = m.angular
        dist_goal = np.linalg.norm(motions, axis=2)
        near_goal = (dist_goal < dist_goal_success).all(1, keepdims=True)
        success = np.logical_and(near_goal, ~collided.any(1, keepdims=True))
        done = success

        reward = np.zeros((achieved_goal.shape[0], 1))
        energy = np.linalg.norm(action, axis=1)[:, None]
        reward[:] = -energy
        reward[~collided] += self.dict_reward["free"]
        reward[collided] += self.dict_reward["collision"]
        reward[success] += self.dict_reward["goal"]

        return reward, done, success

    def batch_compute_rewards(
        self,
        batch_next_achieved_goal,
        batch_goal,
        batch_action,
        collided,
        her_previous_reward=None,
    ):
        reward, done, success = self.compute_rewards(
            batch_next_achieved_goal, batch_goal, batch_action, collided
        )
        return reward, done, success

    def solve_rrt(self, simplify, max_iterations=2000, max_growth=None):
        assert hasattr(self, "robot_props")
        if max_growth is None:
            max_growth = self.robot_props["action_range"]
            if isinstance(max_growth, np.ndarray) and max_growth.shape[0] > 1:
                max_growth = max_growth[0]
        success, path, trees, iterations = solve.solve(
            self, max_growth, max_iterations, simplify
        )
        return success, path, trees, iterations

    def init_viz(self):
        model_wrapper = self.model_wrapper
        if self.viz is None:
            self.viz = Visualizer(self.viz_name, model_wrapper.copy())

    def render(self, *unused_args, **unused_kwargs):
        self.init_viz()
        if not self.showed_goal and self.state is not None:
            utils.display_start_goal(
                self.viz,
                self.robot,
                self.state,
                self.goal_state,
                self.robot_props["dist_goal"],
                self.start_color,
                self.goal_color,
            )
            self.viz.display(self.goal_state)
            color = (1, 0.65, 0, 1)
            self.viz.create_roadmap("path", color=color)
            time.sleep(1)
            self.showed_goal = True
        qw = None
        if self.state is not None:
            qw = self.state
        self.viz.display(qw)
        time.sleep(0.05)

    def get_state(self):
        return {"current": self.state, "goal": self.goal_state}

    def observation(self):
        return self.observe()

    def observe(self):
        return {"observation": np.zeros(0)}

    def normalize(self, x, coordinate_frame):
        if coordinate_frame == "local":
            return (x - self.normalizer_local["mean"]) / self.normalizer_local["std"]
        elif coordinate_frame == "global":
            return (x - self.normalizer_global["mean"]) / self.normalizer_global["std"]
