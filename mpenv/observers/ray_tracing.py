import numpy as np

import numpy as np
import trimesh
from PIL import Image

from gym import spaces
from gym.spaces import Dict

from mpenv.observers.base import BaseObserver
from mpenv.core import utils

NUM_WITNESS_POINTS = {"sphere": 1, "sphere2d": 1, "s_shape": 6}


class RayTracingObserver(BaseObserver):
    def __init__(self, env, n_samples, n_rays, visibility_radius, memory_distance):
        super().__init__(env)

        self.n_samples = n_samples
        self.obstacle_point_dim = 6
        self.obstacles_dim = self.n_samples * self.obstacle_point_dim
        self.n_pts_buffer = 256
        self.visibility_radius = visibility_radius
        self.memory_distance = memory_distance
        self.n_witnesses = NUM_WITNESS_POINTS[env.robot_name]
        self.n_rays_witness = n_rays // self.n_witnesses

        # update observation definition to add the obstacles representation
        self.add_observation("obstacles", self.obstacles_dim)
        self.coordinate_frame = "local"

    def reset(self, **kwargs):
        o = self.env.reset(**kwargs)
        self.geoms = self.env.geoms
        # fixed obstacles
        self.ray_intersector, self.geoms_scene = self.geoms.ray_intersector()
        self.rays = utils.fibonacci_sphere(self.n_rays_witness)
        self.union_pcd = np.zeros((0, 6))
        o = self.observation(o)
        return o

    def represent_obstacles(self, oMi, oMg):
        ref = oMi[1]
        ref_inv = ref.inverse()
        rays_origins = self.witness_points(self.env.robot_name, oMi, oMg)

        origins, rays = self.geoms.compute_origins_rays(rays_origins, self.rays)
        points, normals, _ = self.geoms.ray_intersections(
            self.ray_intersector, self.geoms_scene, origins, rays
        )

        # points in reference frame
        points_ref = utils.apply_transformation(ref_inv, points)
        # only keep points close to reference
        norm_ref = np.linalg.norm(points_ref, axis=1)
        close = np.where(norm_ref < self.visibility_radius)[0]
        points = points[close]
        points_ref = points_ref[close]
        normals = normals[close]
        normals_ref = normals.dot(ref_inv.rotation.T)
        norm_ref = norm_ref[close]

        # local
        indices = np.argsort(norm_ref)
        points_ref = points_ref[indices]
        normals_ref = normals_ref[indices]
        local_pcd = np.hstack((points_ref, normals_ref))
        if local_pcd.shape[0] == 0:
            local_pcd = np.zeros((1, 6))
        sparse_pcd, indices = utils.sparse_subset(local_pcd, self.memory_distance)
        obstacles_repr = sparse_pcd[: self.n_samples]
        obstacles_repr = utils.match_size(sparse_pcd, self.n_samples)[0]

        # uncomment for visualization
        # points, normals = obstacles_repr[:, :3], obstacles_repr[:, 3:]
        # self.env.o3d_viz.show_pcd(points, normals, blocking=False)

        obstacles_repr[:, :3] = self.normalize(
            obstacles_repr[:, :3], self.coordinate_frame
        )

        return obstacles_repr

    def compute_obs(self, state):
        q, oMi, oMg = state.q_oM

        obstacles_pcd = self.represent_obstacles(oMi, oMg)

        return {"pcd": obstacles_pcd.flatten()}

    def observation(self, obs):
        state = self.env.get_state()
        current_state, goal_state = state["current"], state["goal"]
        obs_wrapper = self.compute_obs(current_state)

        obs["observation"] = np.concatenate((obs["observation"], obs_wrapper["pcd"]))

        return obs

    def witness_points(self, robot_name, oMi, oMg):
        if robot_name == "s_shape":
            center = np.array([0.01, 0, 0.01])
            x, y, z = 0.15, 0.075, 0.15
            pts = np.array(
                [[-x, -y, 0], [x, y, 0], [0, -y, 0], [0, y, 0], [x, y, z], [0, y, z]]
            )
            pts += center
        elif robot_name in ["sphere", "sphere2d"]:
            pts = np.zeros((1, 3))
        else:
            raise ValueError(f"Unknown robot: {robot_name}")
        pts = utils.apply_transformation(oMi[1], pts)
        return pts
