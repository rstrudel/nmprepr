import numpy as np

from gym import spaces
from gym.spaces import Dict

from mpenv.observers.base import BaseObserver
from mpenv.core import utils


class PointCloudObserver(BaseObserver):
    def __init__(self, env, n_samples, coordinate_frame, on_surface, add_normals):
        super().__init__(env)

        if not isinstance(add_normals, bool):
            raise ValueError("add_normals is a boolean argument")

        self.n_samples = n_samples
        self.on_surface = on_surface
        if on_surface and add_normals:
            self.obstacle_point_dim = 6
        else:
            self.obstacle_point_dim = 3
        self.add_normals = add_normals
        self.obstacles_dim = self.n_samples * self.obstacle_point_dim
        self.coordinate_frame = coordinate_frame
        if coordinate_frame not in ["local", "global"]:
            raise ValueError(f"Invalid coordinate system: {coordinate_frame}")

        # update observation definition to add the obstacles representation
        self.add_observation("obstacles", self.obstacles_dim)

    def reset(self, **kwargs):
        o = self.env.reset(**kwargs)
        self.obstacles_pcd = self.compute_pcd()
        o = self.observation(o)
        return o

    def compute_pcd(self):
        if self.on_surface:
            points, normals = self.env.compute_surface_pcd(self.n_samples)
            if self.add_normals:
                obstacles_pcd = np.hstack((points, normals))
            else:
                obstacles_pcd = points
        else:
            obstacles_pcd = self.env.compute_volume_pcd(self.n_samples)
        return obstacles_pcd

    def represent_obstacles(self, oMi, obstacles_pcd):
        obstacles_repr = obstacles_pcd.copy()

        ref = oMi[1]
        ref_inv = ref.inverse()
        points, normals = obstacles_repr[:, :3], obstacles_repr[:, 3:]

        if self.coordinate_frame == "local":
            points_ref = utils.apply_transformation(ref_inv, points)
        elif self.coordinate_frame == "global":
            points_ref = points

        if self.add_normals:
            if self.coordinate_frame == "local":
                normals_ref = normals.dot(ref_inv.rotation.T)
            elif self.coordinate_frame == "global":
                normals_ref = normals

            obstacles_repr = np.hstack((points_ref, normals_ref))
        else:
            obstacles_repr = points_ref

        # uncomment for visualization
        # points, normals = obstacles_pcd[:, :3], obstacles_pcd[:, 3:]
        # points = obstacles_pcd[:, :3]
        # normals = np.zeros_like(points)
        # self.env.o3d_viz.show_pcd(points, normals, blocking=False)

        obstacles_repr[:, :3] = self.normalize(
            obstacles_repr[:, :3], self.coordinate_frame
        )

        return obstacles_repr

    def compute_obs(self, state):
        q, oMi, oMg = state.q_oM

        # dynamic obstacles
        # self.obstacles_pcd = self.compute_pcd()
        obstacles_pcd = self.represent_obstacles(oMi, self.obstacles_pcd)
        # self.viz.show_obstacles_pin(self.obstacles_pcd)

        return {"pcd": obstacles_pcd.flatten()}

    def observation(self, obs):
        state = self.env.get_state()
        current_state, goal_state = state["current"], state["goal"]
        obs_wrapper = self.compute_obs(current_state)

        obs["observation"] = np.concatenate((obs["observation"], obs_wrapper["pcd"]))
        return obs

    def show_representation(self):
        fig, ax = self.env.render_matplotlib()
        pcd = self.obstacles_pcd
        X, Y = pcd[:, 0], pcd[:, 1]
        ax.scatter(X, Y, marker="o", c="red", s=40, alpha=0.8)
        if self.add_normals:
            U, V = pcd[:, 3], pcd[:, 4]
            ax.quiver(X, Y, U, V, width=0.005)
        return fig, ax
