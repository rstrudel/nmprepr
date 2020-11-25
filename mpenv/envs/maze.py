import os
import numpy as np
from gym import spaces
import hppfcl
import csv
import pinocchio as pin
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.collections as collections

from mpenv.core.mesh import Mesh
from mpenv.envs.base import Base
from mpenv.envs.maze_generator import Maze
from mpenv.envs import utils as envs_utils
from mpenv.envs.utils import ROBOTS_PROPS
from mpenv.core import utils
from mpenv.core.geometry import Geometries

from mpenv.observers.robot_links import RobotLinksObserver
from mpenv.observers.point_cloud import PointCloudObserver
from mpenv.observers.ray_tracing import RayTracingObserver
from mpenv.observers.maze import MazeObserver


class MazeGoal(Base):
    def __init__(self):
        super().__init__(robot_name="sphere")

        self.thickness = 0.02
        self.grid_size = 7
        self.robot_name = "sphere"
        self.freeflyer_bounds = np.array(
            [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], [1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0]]
        )
        self.robot_props = ROBOTS_PROPS["sphere2d"]
        self.action_space = spaces.Box(
            low=-1, high=1, shape=(self.robot_props["action_dim"],), dtype=np.float32
        )

        self.normalizer_local = {"mean": 0.0, "std": 0.3}
        self.normalizer_global = {"mean": 0.5, "std": 0.5}

    def _reset(self, idx_env=None, start=None, goal=None):
        model_wrapper = self.model_wrapper
        self.robot = self.add_robot("sphere2d", self.freeflyer_bounds)
        self.geoms, self.idx_env = self.get_obstacles_geoms(idx_env)
        for geom_obj in self.geoms.geom_objs:
            self.add_obstacle(geom_obj, static=True)
        model_wrapper.create_data()

        valid_sample = False
        while not valid_sample:
            self.state = self.random_configuration()
            self.goal_state = self.random_configuration()
            valid_sample = True
        if start is not None:
            self.set_state(start)
        if goal is not None:
            self.set_goal_state(goal)

        return self.observation()

    def get_obstacles_geoms(self, idx_env):
        np_random = self._np_random
        self.maze = Maze(self.grid_size, self.grid_size)
        self.maze.make_maze()
        geom_objs = extract_obstacles(self.maze, self.thickness)
        geoms = Geometries(geom_objs)
        return geoms, idx_env

    def set_eval(self):
        pass


def extract_obstacles(maze, thickness):
    scx = 1 / maze.nx
    scy = 1 / maze.ny

    obstacles_coord = []
    obstacles_coord.append((0, 0, 1, 0))
    obstacles_coord.append((0, 0, 0, 1))
    obstacles_coord.append((1, 0, 1, 1))
    obstacles_coord.append((0, 1, 1, 1))
    # Draw the "South" and "East" walls of each cell, if present (these
    # are the "North" and "West" walls of a neighbouring cell in
    # general, of course).
    for x in range(maze.nx - 1):
        for y in range(maze.ny - 1):
            if maze.cell_at(x, y).walls["S"]:
                x1, y1, x2, y2 = (
                    x * scx,
                    (y + 1) * scy,
                    (x + 1) * scx,
                    (y + 1) * scy,
                )
                obstacles_coord.append((x1, y1, x2, y2))
            if maze.cell_at(x, y).walls["E"]:
                x1, y1, x2, y2 = (
                    (x + 1) * scx,
                    y * scy,
                    (x + 1) * scx,
                    (y + 1) * scy,
                )
                obstacles_coord.append((x1, y1, x2, y2))
    obstacles = []
    for i, obst_coord in enumerate(obstacles_coord):
        x1, y1, x2, y2 = obst_coord[0], obst_coord[1], obst_coord[2], obst_coord[3]
        if np.allclose(x1, x2):
            x1 -= thickness / 2
            x2 = x1 + thickness
        if np.allclose(y1, y2):
            y1 -= thickness / 2
            y2 = y1 + thickness
        box_size = [x2 - x1, y2 - y1, 0.1]
        pos = [(x1 + x2) / 2, (y1 + y2) / 2, 0]
        placement = pin.SE3(np.eye(3), np.array(pos))
        mesh = Mesh(
            name=f"obstacle{i}",
            geometry=hppfcl.Box(*box_size),
            placement=placement,
            color=(0, 0, 1, 0.8),
        )
        obstacles.append(mesh.geom_obj())

    return obstacles


def maze_pointcloud(n_samples, on_surface, add_normals, coordinate_frame):
    env = MazeGoal()
    env = PointCloudObserver(env, n_samples, coordinate_frame, on_surface, add_normals)
    env = RobotLinksObserver(env, coordinate_frame)
    return env


def maze_edges():
    env = MazeGoal()
    env = MazeObserver(env)
    coordinate_frame = "local"
    env = RobotLinksObserver(env, coordinate_frame)
    return env


def maze_raytracing(n_samples, n_rays):
    env = MazeGoal()
    visibility_radius = 0.7
    memory_distance = 0.06
    env = RayTracingObserver(env, n_samples, n_rays, visibility_radius, memory_distance)
    coordinate_frame = "local"
    env = RobotLinksObserver(env, coordinate_frame)
    return env
