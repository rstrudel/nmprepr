from itertools import cycle

import numpy as np

import eigenpy
import hppfcl
import pinocchio as pin
from pinocchio.visualize import GepettoVisualizer, MeshcatVisualizer
from mpenv.core.mesh import Mesh


class Visualizer:
    def __init__(self, name, model_wrapper):
        self.name = name
        if self.name == "meshcat":
            self.viz_class = MeshcatVisualizer
        elif self.name == "gepetto":
            self.viz_class = GepettoVisualizer
        else:
            raise ValueError(f"Unknown visualizer: {self.name}")

        self.model_wrapper = model_wrapper
        self.model_wrapper.create_data()
        # self.viewer = None
        self.node_name = "core"

        self._create_viz()

    def _create_viz(self):
        model = self.model_wrapper.model
        geom_model = self.model_wrapper.geom_model
        self.viz = self.viz_class(model, geom_model, geom_model)
        self.viz.initViewer()

        node_name = self.node_name
        if self.name == "gepetto":
            self.gui = self.viz.viewer.gui
            gui = self.gui
            self.window_id = gui.getWindowList()[0]
            if gui.nodeExists(f"world/{node_name}"):
                gui.deleteNode(f"world/{node_name}", True)
                gui.setBackgroundColor1(self.window_id, (1, 1, 1, 1))
                gui.setBackgroundColor2(self.window_id, (1, 1, 1, 0.5))
            # gui.setLightingMode("world", "OFF")
            # gui.setLightingMode("world", "ON")
        self.viz.loadViewerModel(node_name)

    def _update_data(self):
        self.model_wrapper.create_data()
        self._create_viz()

    def display(self, qw=None):
        if qw is None:
            qw = self.model_wrapper.neutral_configuration()
        q = qw.q
        self.viz.display(q)

    def add_geom_obj(self, geom_obj, update_data=True):
        geom_model = self.model_wrapper.geom_model
        geom_model.addGeometryObject(geom_obj)
        if update_data:
            self._update_data()

    def show_bounds(self, bounds):
        pos = bounds.mean(axis=0)
        size = (bounds[1] - bounds[0]).astype(float)
        workspace = Mesh(
            name="workspace",
            geometry=hppfcl.Box(*size),
            placement=pin.SE3(np.eye(3), pos),
            color=(0, 0, 1, 0.5),
        )
        self.add_mesh(workspace, update_data=True)

    def show_joints(self):
        raise ValueError("To be reimplemented")
        goal_jts = utils.get_oMi(self.model, self.data, self.goal_state["q"])
        for i, jt_se3 in enumerate(goal_jts):
            self.add_mesh(
                f"jt{i}",
                geom=hppfcl.Sphere(0.07),
                placement=jt_se3,
                check_collision=False,
                color=(0, 0, 1, 1.0),
            )
        self._create_data()
        self._create_viz()

    def show_robot_aabb(self):
        def hex_to_rgb(value):
            value = value.lstrip("#")
            lv = len(value)
            return tuple(int(value[i : i + lv // 3], 16) for i in range(0, lv, lv // 3))

        # geom_objs = self.geom_model.geometryObjects[1:7]
        geom_objs = self.geom_model.geometryObjects
        geometries = [geom_obj.geometry for geom_obj in geom_objs]
        parents = [geom_obj.parentJoint for geom_obj in geom_objs]
        i = 0
        colors = cycle(
            [
                "#377eb8",
                "#e41a1c",
                "#4daf4a",
                "#984ea3",
                "#ff7f00",
                "#ffff33",
                "#a65628",
                "#f781bf",
            ]
        )
        for geom, parent_id, color in zip(geometries, parents, colors):
            aabb = geom.aabb_local
            w, h, d = aabb.width(), aabb.height(), aabb.depth()
            box = hppfcl.Box(w, h, d)
            placement = pin.SE3(np.eye(3), aabb.center())
            geom_obj = pin.GeometryObject(f"aabb{i}", 0, parent_id, box, placement)
            color = np.array(hex_to_rgb(color)) / 255
            geom_obj.meshColor = np.array((color[0], color[1], color[2], 0.5))
            self.viz_model.addGeometryObject(geom_obj)
            i += 1
        print("show aabb")
        self._create_data()
        self._create_viz()

    def show_obstacles_pin(self, obstacles):
        for i, sample in enumerate(obstacles):
            rot = eigenpy.Quaternion.FromTwoVectors(np.array((0, 0, 1)), sample[3:])
            rot = rot.toRotationMatrix()
            cone = Mesh(
                name=f"surf{i}",
                geometry=hppfcl.Cone(0.03, 0.1),
                placement=pin.SE3(rot, sample[:3]),
                color=(0, 0, 0, 0.8),
            )
            self.add_mesh(cone, update_data=False)
        self._update_data()

    def create_roadmap(self, name, color):
        if not self.name == "gepetto":
            raise ValueError("Only implemented for gepetto-gui")

        roadmap_name = f"world/{self.node_name}/{name}"
        self.gui.createRoadmap(
            roadmap_name, (0, 0, 0, 1), 1, 1, color,
        )

    def add_edge_to_roadmap(self, name, start, end):
        roadmap_name = f"world/{self.node_name}/{name}"
        self.gui.addEdgeToRoadmap(roadmap_name, list(start), list(end))

    def display_tree(self, nodes, name, color, create_roadmap=True):
        if create_roadmap:
            self.create_roadmap(name, color=color)
        drawn_edges = []
        for node in nodes:
            edge = None
            if node.parent:
                edge = (node.parent.point[:3], node.point[:3])
                edge = tuple(map(tuple, edge))
                if edge not in drawn_edges:
                    self.add_edge_to_roadmap(name, edge[0], edge[1])
                    drawn_edges.append(edge)
