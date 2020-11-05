import numpy as np

import hppfcl
import pinocchio as pin
from mpenv.core.model import ModelWrapper
from mpenv.robot.robot import Robot
from mpenv.core.mesh import Mesh
from mpenv.core import utils


class Sphere:
    def __init__(self, model, geom_model, name, radius, bounds, color=(1, 0, 0, 1)):
        self.radius = radius
        self.color = color
        self.geom_obj = self.make_geom_obj(name)
        bounds_ori = np.zeros((2, 7))
        bounds_ori[:, :3] = bounds
        bounds_ori[:, 3:6] = 0
        bounds_ori[:, -1] = 1
        self.bounds = bounds_ori
        self.model = model
        self.geom_model = geom_model
        self.n_joints = 1
        self.q_dim = 7
        self.link_dim = 3
        self.n_witnesses = 1

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
        return oMg[0].translation

    def get_ee(self, oMg):
        return oMg[0]

    def project_q(self, q):
        return np.clip(q, self.bounds[0], self.bounds[1])

    def make_geom_obj(self, name):
        geom = hppfcl.Sphere(self.radius)
        mesh = Mesh(name=name, geometry=geom, color=self.color)
        return mesh.geom_obj()
