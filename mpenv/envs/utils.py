import os
import pickle as pkl
import numpy as np
import hppfcl
import pinocchio as pin
import eigenpy

from mpenv.core.mesh import Mesh
from mpenv.core.geometry import Geometries


LINVEL_RANGE = np.array([0.07, 0.07, 0.07])
ANGVEL_RANGE = np.array([0.2, 0.2, 0.2])
VEL_RANGE = np.hstack((LINVEL_RANGE, ANGVEL_RANGE))

ROBOTS_PROPS = {
    "sphere": {
        "dist_goal": 0.07,
        "action_dim": 3,
        "action_range": LINVEL_RANGE[0],
        "local": {"link_dim": 3, "goal_dim": 16, "goal_rep_dim": 3},
        "global": {"link_dim": 3, "goal_dim": 16, "goal_rep_dim": 3},
        "n_joints": 1,
        "link_action_dim": 3,
    },
    "sphere2d": {
        "dist_goal": 0.07,
        "action_dim": 2,
        "action_range": LINVEL_RANGE[0],
        "local": {"link_dim": 2, "goal_dim": 16, "goal_rep_dim": 2},
        "global": {"link_dim": 2, "goal_dim": 16, "goal_rep_dim": 2},
        "n_joints": 1,
        "link_action_dim": 2,
    },
    "s_shape": {
        "dist_goal": 0.07,
        "action_dim": 6,
        "action_range": VEL_RANGE,
        "local": {"link_dim": 7, "goal_dim": 16, "goal_rep_dim": 6},
        "global": {"link_dim": 7, "goal_dim": 7, "goal_rep_dim": 7},
        "n_joints": 1,
        "link_action_dim": 6,
    },
}


def load_dataset_geoms(filename):
    with open(filename, "rb") as f:
        geoms_pkl = pkl.load(f)
    dataset_geoms = []
    for geoms_dict in geoms_pkl["geoms_dicts"]:
        geoms = Geometries()
        geoms.from_dict(geoms_dict)
        dataset_geoms.append(geoms)
    return dataset_geoms


def display_start_goal(
    viz, robot, state, goal_state, dist_goal, start_color, goal_color
):
    if viz is None:
        raise ValueError("No visualizer instantiated.")
    start_oMg = state.oMg
    goal_oMg = goal_state.oMg
    start_oMg_np = robot.get_oMg_np(start_oMg)
    goal_oMg_np = robot.get_oMg_np(goal_oMg)
    # display start and goal ee pos
    for i, (start_i, goal_i) in enumerate(zip(start_oMg_np, goal_oMg_np)):
        start_robot = robot.make_geom_obj(f"start{i}", i)
        goal_robot = robot.make_geom_obj(f"goal{i}", i)
        start_robot.placement = pin.SE3(start_i) * start_robot.placement
        start_robot.meshColor = start_color
        goal_robot.placement = pin.SE3(goal_i) * goal_robot.placement
        goal_robot.meshColor = goal_color
        for geom_obj in [start_robot, goal_robot]:
            viz.add_geom_obj(geom_obj)


def get_bounds_geom_objs(pos_bounds):
    """
    Generate 6 faces corresponding to the agent deplacement bounds
    """
    size = pos_bounds[1] - pos_bounds[0]
    center = np.mean(pos_bounds, axis=0)
    thickness = 0.05
    color = (1, 1, 1, 0.3)
    geom_objs = []
    aas = [
        eigenpy.AngleAxis(0, np.array([1, 0, 0])),
        eigenpy.AngleAxis(np.pi / 2, np.array([0, 1, 0])),
        eigenpy.AngleAxis(np.pi / 2, np.array([0, 0, 1])),
    ]
    placement = pin.SE3.Identity()
    for i, angle_axis in enumerate(aas):
        placement.rotation = angle_axis.matrix()
        size_bound = size.copy()
        translation = np.zeros(3)
        size_bound[i] = thickness
        translation[i] = -size[i] / 2 - thickness / 2.1
        for j in range(2):
            geom = hppfcl.Box(*size_bound)
            placement.translation = translation
            mesh = Mesh(
                name=f"bound{i}{j}", geometry=geom, placement=placement, color=color,
            )
            geom_objs.append(mesh.geom_obj())
            translation *= -1
    return geom_objs
