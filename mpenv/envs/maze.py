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

    def compute_surface_pcd(self, n_pts):
        return obstacles_to_surface_2dpcd(self.geoms, n_pts)

    def compute_volume_pcd(self, n_pts):
        return obstacles_to_volume_2dpcd(self.geoms, n_pts)

    def compute_occupancy_grid(self, size):
        def collision(pts):
            collisions = np.zeros(pts.shape[0])
            for i, pt in enumerate(pts):
                if 0 <= pt[0] <= 1 and 0 <= pt[1] <= 1:
                    q = np.zeros(7)
                    q[-1] = 1
                    q[:2] = pt
                    collisions[i] = self.model_wrapper.collision(q)
                else:
                    collisions[i] = True

            return collisions

        return obstacles_to_occupancy_grid(size, collision)

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


def obstacles_to_surface_2dpcd(geoms, n_pts):
    points, normals = geoms.compute_surface_pcd(5 * n_pts)
    # reject points with normals in the z direction or outside of [0, 1]
    within_box = np.logical_and(points[:, :2] < 1.0, points[:, :2] > 0.0).all(axis=1)
    hori_normals = np.abs(normals[:, 2]) < 1e-3
    valid_pts = np.logical_and(within_box, hori_normals)
    points = points[valid_pts]
    normals = normals[valid_pts]
    points[:, 2] = 0
    # sparse representation
    points, indices = utils.sparse_subset(points, 0.015)
    normals = normals[indices]
    # match required size
    points, indices = utils.match_size(points, n_pts)
    normals = normals[indices]
    return points, normals


def obstacles_to_volume_2dpcd(geoms, n_pts):
    points = geoms.compute_volume_pcd(int(1.5 * n_pts))
    points[:, 2] = 0
    # sparse representation
    points, indices = utils.sparse_subset(points, 0.015)
    # match required size
    points, indices = utils.match_size(points, n_pts)
    return points


def obstacles_to_occupancy_grid(grid_size, collision):
    cond_occ = np.zeros((grid_size * grid_size))
    occ_grid_samples = np.zeros((grid_size * grid_size, 2))
    grid_points_range = np.linspace(0, 1, num=grid_size + 1)[:-1] + 1 / (2 * grid_size)

    idx = 0
    for j, y in enumerate(grid_points_range):
        for i, x in enumerate(grid_points_range):
            # cond_occ_xy[1:, j, i] = np.array([x, y])
            occ_grid_samples[idx, 0] = x
            occ_grid_samples[idx, 1] = y
            idx += 1

    occ_grid = collision(occ_grid_samples)

    return occ_grid, occ_grid_samples


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
