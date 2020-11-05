import os
import numpy as np
from gym import spaces
import pinocchio as pin
import hppfcl

from mpenv.envs.base import Base
from mpenv.envs import utils as envs_utils
from mpenv.envs.utils import ROBOTS_PROPS
from mpenv.core import utils
from mpenv.core.geometry import Geometries
from mpenv.core.mesh import Mesh
from mpenv.core.model import ConfigurationWrapper

from mpenv.observers.robot_links import RobotLinksObserver
from mpenv.observers.point_cloud import PointCloudObserver
from mpenv.observers.ray_tracing import RayTracingObserver


class Boxes(Base):
    def __init__(
        self,
        robot_name,
        has_boxes,
        cube_bounds=True,
        obstacles_type="boxes",
        dynamic_obstacles=False,
    ):
        super().__init__(robot_name)

        self.has_boxes = has_boxes
        self.geoms = None
        self.cube_bounds = cube_bounds
        self.obstacles_type = obstacles_type
        self.dynamic_obstacles = dynamic_obstacles

        self.robot_props = ROBOTS_PROPS[self.robot_name]
        self._set_obstacles_props()
        self.action_space = spaces.Box(
            low=-1, high=1, shape=(self.robot_props["action_dim"],), dtype=np.float32
        )

        self.normalizer_local = {"mean": 0, "std": 0.4}
        self.normalizer_global = {"mean": 0, "std": 0.5}

    def _set_obstacles_props(self):
        if self.robot_name == "s_shape":
            self.freeflyer_bounds = np.array(
                [
                    [-0.6, -0.6, -0.6, -np.inf, -np.inf, -np.inf, -np.inf],
                    [0.6, 0.6, 0.6, np.inf, np.inf, np.inf, np.inf],
                ]
            )
        else:
            self.freeflyer_bounds = np.array([[-0.6, -0.6, -0.6], [0.6, 0.6, 0.6]])

        # constrain box xyz position to not be on boundary to get a denser obstacle set
        self.center_bounds = self.freeflyer_bounds[:, :3].copy()
        self.center_bounds[0, :3] += np.array([0.15, 0.15, 0.15])
        self.center_bounds[1, :3] -= np.array([0.15, 0.15, 0.15])
        self.center_bounds
        self.size_bounds = [0.15, 0.5]

    def _reset(self, start=None, goal=None):
        self.geoms = self.get_obstacles_geoms()
        self.robot = self.add_robot(self.robot_name, self.freeflyer_bounds)

        for geom_obj in self.geoms.geom_objs:
            self.add_obstacle(geom_obj, static=True)
        self.model_wrapper.create_data()

        if start is not None:
            if not isinstance(start, ConfigurationWrapper):
                start = ConfigurationWrapper(self.model_wrapper, start)
            self.state = start
        else:
            self.state = self.random_configuration()
        if goal is not None:
            if not isinstance(goal, ConfigurationWrapper):
                goal = ConfigurationWrapper(self.model_wrapper, goal)
            self.goal_state = goal
        else:
            self.goal_state = self.random_configuration()

        return self.observation()

    def get_obstacles_geoms(self):
        if not self.has_boxes:
            return Geometries()

        # boxes
        if self.n_obstacles is None:
            n_obstacles = self._np_random.randint(3, 10)
        else:
            n_obstacles = self.n_obstacles

        geom_objs, placement_tuple = generate_geom_objs(
            self._np_random,
            self.freeflyer_bounds,
            self.center_bounds,
            self.size_bounds,
            self.cube_bounds,
            n_obstacles,
            self.obstacles_type,
            self.dynamic_obstacles,
            self.obstacles_color,
            self.obstacles_alpha,
        )
        self.se3_obst_tuple = placement_tuple
        geoms = Geometries(geom_objs)
        return geoms

    def compute_surface_pcd(self, n_pts):
        return obstacles_to_surface_pcd(self.geoms, n_pts, self.freeflyer_bounds)

    def compute_volume_pcd(self, n_pts):
        return self.geoms.compute_volume_pcd(n_pts)

    def set_eval(self):
        pass


def sample_box_parameters(np_random, center_bounds, size_bounds):
    box_size = np_random.uniform(size_bounds[0], size_bounds[1], size=3)
    pos_box = np_random.uniform(center_bounds[0], center_bounds[1], 3)
    rand_se3 = pin.SE3.Identity()
    pin.SE3.setRandom(rand_se3)
    rand_se3.translation = pos_box
    return rand_se3, box_size


def sample_geom(np_random, obst_type, size, index=None):
    geom = None
    path = ""
    scale = np.ones(3)
    if obst_type != "boxes":
        size /= 2

    if obst_type == "boxes":
        geom = hppfcl.Box(*size)
    elif obst_type == "shapes":
        j = np.random.randint(3)
        if j == 0:
            r = 1.3 * size[0]
            geom = hppfcl.Sphere(r)
        elif j == 1:
            geom = hppfcl.Cylinder(size[0] / 1.5, size[1] * 3)
        elif j == 2:
            # geom = hppfcl.Cone(size[0] * 1.5, 3 * size[1])
            geom = hppfcl.Capsule(size[0], 2 * size[1])
    elif obst_type == "ycb":
        dataset_path = "YCB_PATH"
        files = os.listdir(dataset_path)
        idx = np.random.randint(len(files))
        path = os.path.join(dataset_path, files[idx])
        scale = 3.5 * np.ones(3)
    return geom, path, scale


def generate_geom_objs(
    np_random,
    freeflyer_bounds,
    center_bounds,
    size_bounds,
    cube_bounds,
    n_obstacles,
    obstacles_type,
    dynamic_obstacles,
    obstacles_color,
    obstacles_alpha,
):
    colors = np_random.uniform(0, 1, (n_obstacles, 4))
    colors[:, 3] = obstacles_alpha
    if obstacles_color is not None:
        colors = [obstacles_color for _ in range(n_obstacles)]
    name = "box{}"
    geom_objs = []
    placement_tuple = []
    # obstacles
    for i in range(n_obstacles):
        rand_se3_init, obst_size = sample_box_parameters(
            np_random, center_bounds, size_bounds
        )
        rand_se3_target, obst_size = sample_box_parameters(
            np_random, center_bounds, size_bounds
        )
        placement_tuple.append((rand_se3_init, rand_se3_target))
        # rand_se3.rotation = np.eye(3)
        geom, path, scale = sample_geom(np_random, obstacles_type, obst_size, i)
        mesh = Mesh(
            name=name.format(i),
            geometry=geom,
            placement=rand_se3_init,
            color=colors[i],
            geometry_path=path,
            scale=scale,
        )
        geom_obj_obstacle = mesh.geom_obj()
        geom_objs.append(geom_obj_obstacle)
    if cube_bounds:
        geom_objs_bounds = envs_utils.get_bounds_geom_objs(freeflyer_bounds[:, :3])
        geom_objs += geom_objs_bounds
    return geom_objs, placement_tuple


def obstacles_to_surface_pcd(geoms, n_pts, bounds):
    points, normals = geoms.compute_surface_pcd(n_pts)
    return points, normals


def boxes_noobst(robot_name):
    env = Boxes(robot_name, has_boxes=False, cube_bounds=True)
    # coordinate_frame = "global"
    coordinate_frame = "local"
    env = RobotLinksObserver(env, coordinate_frame)
    return env


def boxes_pointcloud(robot_name, n_samples, on_surface, add_normals):
    env = Boxes(robot_name, has_boxes=True, cube_bounds=False, dynamic_obstacles=False)
    coordinate_frame = "local"
    env = PointCloudObserver(env, n_samples, coordinate_frame, on_surface, add_normals)
    env = RobotLinksObserver(env, coordinate_frame)
    return env


def boxes_raytracing(robot_name, n_samples, n_rays):
    env = Boxes(robot_name, has_boxes=True, cube_bounds=True, dynamic_obstacles=False)
    visibility_radius = 0.7
    memory_distance = 0.06
    env = RayTracingObserver(env, n_samples, n_rays, visibility_radius, memory_distance)
    coordinate_frame = "local"
    env = RobotLinksObserver(env, coordinate_frame)
    return env
