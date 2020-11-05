import numpy as np
import os

import hppfcl
import open3d as o3d
import pinocchio as pin


class Mesh:
    def __init__(
        self,
        name,
        geometry=None,
        geometry_path="",
        placement=None,  # 4x4 matrix
        color=(1.0, 0.0, 0.0, 1.0),
        scale=(1.0, 1.0, 1.0),
    ):
        super().__init__()

        if placement is None:
            placement = pin.SE3.Identity()
        assert isinstance(
            placement, pin.SE3
        ), "Use pin.SE3(R, t) with R 3x3, t 3 to define a placement"

        self.name = name
        if geometry:
            self.geometry = geometry
        if geometry_path:
            current_dir = os.path.dirname(__file__)
            geometry_path = os.path.realpath(os.path.join(current_dir, geometry_path))
            meshloader = hppfcl.MeshLoader()
            self.geometry = meshloader.load(
                geometry_path, scale=np.array(scale, dtype=float)
            )
        self.geometry_path = geometry_path
        self.placement = placement
        self.color = np.array(color)
        self.scale = np.array(scale)

    def geom_obj(self, parent_frame=0, parent_joint=0, name=None):
        if name is None:
            name = self.name
        geom_obj = pin.GeometryObject(
            name=name,
            parent_joint=parent_joint,
            parent_frame=parent_frame,
            collision_geometry=self.geometry,
            placement=self.placement,
            mesh_path=self.geometry_path,
            mesh_scale=self.scale,
            override_material=True,
            mesh_color=self.color,
        )
        return geom_obj
