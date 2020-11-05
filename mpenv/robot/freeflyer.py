import os
import numpy as np
import pinocchio as pin
import hppfcl

from mpenv.core.model import ModelWrapper
from mpenv.robot.robot import Robot

# from mpenv.core.mesh import Mesh
from mpenv.core.geometry import Geometries
from mpenv.core import utils


class FreeFlyer(Robot):
    def __init__(self, model_wrapper, mesh, bounds):

        self.n_joints = 1
        self.q_dim = 7
        self.n_witnesses = 8

        self.mesh = mesh
        if bounds is None:
            bounds_ori = bounds
        elif bounds.shape[1] == 3:
            bounds_ori = np.zeros((2, 7))
            bounds_ori[:, :3] = bounds
            bounds_ori[:, 3:6] = 0
            bounds_ori[:, 6] = 1
        elif bounds.shape[1] == 7:
            bounds_ori = bounds
        else:
            raise ValueError(
                f"bounds is of shape {bounds.shape} instead of (2, 3) or (2, 7)."
            )
        self.bounds = bounds_ori

        geom_obj = mesh.geom_obj()
        self._build_from_mesh(model_wrapper, mesh, bounds_ori)

    def get_representation(self, q, oMi, oMg):
        """
        parent joint [X, Y, Z, Goal] one hot encoding
        parent theta (controlled by the body)
        child joint [X, Y, Z, Goal] one hot encoding
        child theta
        body placement [R, t]
        body dimension (width, height, depth)
        goal (gx, gy, gz)
        """
        return q

    def make_geom_obj(self, name=None, unused_index=None):
        return self.mesh.geom_obj(name=name)

    def get_ee(self, oMg):
        return oMg[0]

    def get_oMg_np(self, oMg):
        oMg_np = oMg[0].np[None, :]
        return oMg_np
