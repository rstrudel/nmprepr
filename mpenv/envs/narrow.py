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
from mpenv.envs import utils as envs_utils
from mpenv.envs.utils import ROBOTS_PROPS
from mpenv.core import utils
from mpenv.core.geometry import Geometries

from mpenv.observers.robot_links import RobotLinksObserver
from mpenv.observers.point_cloud import PointCloudObserver
from mpenv.observers.corners import CornersObserver
from mpenv.observers.image import ImageObserver


class NarrowGoal(Base):
    def __init__(self, max_env_idx=None):
        super().__init__(robot_name="sphere")

        self.num_obstacles = 5
        self.max_env_idx = max_env_idx
        self.grid_size = 10
        self.robot_name = "sphere"
        self.freeflyer_bounds = np.array(
            [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], [1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0]]
        )
        self.robot_props = ROBOTS_PROPS["sphere2d"]
        self.action_space = spaces.Box(
            low=-1, high=1, shape=(self.robot_props["action_dim"],), dtype=np.float32
        )

        self.x, self.c = load_obstacles_from_data(train=True)

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

    def validate_sample(self, state, goal_state):
        "Filter start and goal with straight path solution"
        straight_path = self.model_wrapper.arange(
            state, goal_state, self.delta_collision_check
        )
        _, collide = self.stopping_configuration(straight_path)
        return collide

    def get_obstacles_geoms(self, idx_env):
        np_random = self._np_random
        if idx_env is None:
            if self.max_env_idx is None:
                idx_env = np_random.randint(self.c.shape[0])
            else:
                idx_env = np_random.randint(self.max_env_idx)
        # idx env used for paper figures
        # idx_env = 14359

        c = self.c[idx_env]
        geom_objs = extract_obstacles(c, self.num_obstacles)
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
        self.x, self.c = load_obstacles_from_data(train=False)

    def render_matplotlib(self):
        fig = plt.figure(figsize=(10, 6), dpi=80)
        ax = fig.add_subplot(aspect="equal")
        obstacles = self.geoms.geom_objs
        rects = []
        for i, obst in enumerate(obstacles):
            x, y = obst.placement.translation[:2]
            half_side = obst.geometry.halfSide
            w, h = 2 * half_side[:2]
            rects.append(
                patches.Rectangle(
                    (x - w / 2, y - h / 2), w, h  # (x,y)  # width  # height
                )
            )
        coll = collections.PatchCollection(rects, zorder=1)
        coll.set_alpha(0.6)
        ax.add_collection(coll)
        init = self.state.q
        goal = self.goal_state.q
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.0)
        ax.set_xticks([])
        ax.set_yticks([])
        return fig, ax


def load_obstacles_from_data(train):
    csv_filename = os.path.join(os.path.dirname(__file__), "../assets/narrow_data.csv")
    ratio_train = 0.8
    # problem dimension
    state_dim = 6

    w_dim = 3
    num_gaps = 3
    # sample (6D), gap1 (2D, 1D orientation), gap2, gap3, init (6D), goal (6D)
    dim_data = state_dim + num_gaps * w_dim + 2 * state_dim
    # dimension of conditioning variable
    data = read_csv(csv_filename, dim_data)
    n_train = int(data.shape[0] * ratio_train)
    # state: x, y, z, xdot, ydot, zdot
    x = data[:, :state_dim]
    # cond: gaps, init (6), goal (6). only keep gaps
    c = data[:, state_dim : state_dim + 3 * w_dim]
    if train:
        x, c = x[:n_train], c[:n_train]
    else:
        x, c = x[n_train:], c[n_train:]
    return x, c


def read_csv(filename, dim_data):
    count = 0
    data_list = []
    with open(filename, "r") as f:
        reader = csv.reader(f, delimiter=",")
        for row in reader:
            data_list.append(list(map(float, row[:dim_data])))
            count += 1
    data = np.array(data_list, dtype="d")
    return data


def extract_obstacles(condition, num_obstacles):
    w_dim = 3
    dw = 0.1
    gap1 = condition[0:w_dim]
    gap2 = condition[w_dim : 2 * w_dim]
    gap3 = condition[2 * w_dim : 3 * w_dim]

    num_obstacles = 5
    obst1 = [0, gap1[1] - dw, -0.5, gap1[0], gap1[1], 1.5]
    obst2 = [gap2[0] - dw, 0, -0.5, gap2[0], gap2[1], 1.5]
    obst3 = [gap2[0] - dw, gap2[1] + dw, -0.5, gap2[0], 1, 1.5]
    obst4 = [gap1[0] + dw, gap1[1] - dw, -0.5, gap3[0], gap1[1], 1.5]
    obst5 = [gap3[0] + dw, gap1[1] - dw, -0.5, 1, gap1[1], 1.5]
    obstacles = [obst1, obst2, obst3, obst4, obst5]
    obstacles = obstacles[:num_obstacles]
    for i, obst in enumerate(obstacles):
        x0, y0, x1, y1 = obst[0], obst[1], obst[3], obst[4]
        box_size = [x1 - x0, y1 - y0, 0.1]
        pos = [(x0 + x1) / 2, (y0 + y1) / 2, 0]
        placement = pin.SE3(np.eye(3), np.array(pos))
        mesh = Mesh(
            name=f"obstacle{i}",
            geometry=hppfcl.Box(*box_size),
            placement=placement,
            color=(0, 0, 1, 0.8),
        )
        obstacles[i] = mesh.geom_obj()

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


def narrow_pointcloud(
    max_env_idx, n_samples, on_surface, add_normals, coordinate_frame
):
    env = NarrowGoal(max_env_idx)
    env = PointCloudObserver(env, n_samples, coordinate_frame, on_surface, add_normals)
    env = RobotLinksObserver(env, coordinate_frame)
    return env


def narrow_corners(max_env_idx):
    env = NarrowGoal(max_env_idx)
    coordinate_frame = "local"
    env = CornersObserver(env, coordinate_frame)
    env = RobotLinksObserver(env, coordinate_frame)
    return env


def narrow_image(max_env_idx, size, pov):
    env = NarrowGoal(max_env_idx)
    visibility_distance = 0.5
    env = ImageObserver(env, size, pov, visibility_distance)
    env = RobotLinksObserver(env, coordinate_frame=pov)
    return env
