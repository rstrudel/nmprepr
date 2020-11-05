import numpy as np
import trimesh
import logging

from mpenv.core import utils

trimesh.util.attach_to_log(logging.ERROR)


class Geometries:
    def __init__(self, geom_objs=None):
        if geom_objs is None:
            geom_objs = []
        self.geom_objs = geom_objs
        self.union_mesh = None

    def from_dict(self, state):
        self.union_mesh = state["mesh"]
        geom_obj_dicts = state["geom_props"]
        self.geom_objs = []
        for geom_obj_dict in geom_obj_dicts:
            geom_obj = utils.dict_to_geom_obj(geom_obj_dict)
            self.geom_objs.append(geom_obj)

    def to_dict(self):
        if self.union_mesh is None:
            self.compute_union_mesh()
        state = {"geom_props": [], "mesh": self.union_mesh}
        for geom_obj in self.geom_objs:
            state["geom_props"].append(utils.geom_obj_to_dict(geom_obj))
        return state

    def compute_meshs(self):
        meshs = []
        for geom_obj in self.geom_objs:
            meshs.append(utils.mesh_from_geometry(geom_obj))
        return meshs

    def scene(self):
        scene = trimesh.Scene()
        meshs = self.compute_meshs()
        vertices = np.zeros((0, 3))
        tris = np.zeros((0, 3, 3))
        faces = np.zeros((0, 3), dtype=int)
        for mesh in meshs:
            scene.add_geometry(mesh)
            n = vertices.shape[0]
            vertices = np.vstack((vertices, mesh.vertices))
            tris = np.vstack((tris, mesh.triangles))
            faces = np.vstack((faces, mesh.faces + n))
        # compute face normals
        # scene.triangles = tris
        scene.vertices = vertices
        scene.faces = faces
        normals = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
        area_faces = np.linalg.norm(normals, axis=-1)
        unit_normals = normals / area_faces[:, None]
        scene.face_normals = unit_normals
        scene.area_faces = area_faces
        return scene

    def compute_union_mesh(self):
        if len(self.geom_objs) == 0:
            return None

        meshs = self.compute_meshs()
        # costly function, takes ~0.1sec to 1sec to compute
        if len(meshs) > 1:
            print("computed union")
            mesh = trimesh.boolean.union(meshs)
        else:
            mesh = meshs[0]

        self.union_mesh = mesh

    def compute_origins_rays(self, rays_origins, rays):
        origins = []
        for point in rays_origins:
            origins += [point] * rays.shape[0]
        origins = np.array(origins)
        rays = np.tile(rays, (rays_origins.shape[0], 1))
        return origins, rays

    def ray_intersector(self):
        scene = self.scene()
        return trimesh.ray.ray_pyembree.RayMeshIntersector(scene), scene

    def ray_intersections(self, ray_intersector, scene, origins, rays):
        points, ray_indices, tri_indices = ray_intersector.intersects_location(
            origins, rays, multiple_hits=False
        )
        normals = scene.face_normals[tri_indices]
        return points, normals, ray_indices

    def sample_uniformly_triangle(self, tris):
        # https://math.stackexchange.com/questions/18686/uniform-random-point-in-triangle
        a, b, c = tris[:, 0], tris[:, 1], tris[:, 2]
        r = np.random.uniform(0, 1, size=(tris.shape[0], 2))
        r1, r2 = r[:, 0], r[:, 1]
        r1, r2 = r1[:, None], r2[:, None]
        p = (1 - np.sqrt(r1)) * a + (np.sqrt(r1) * (1 - r2)) * b + r2 * np.sqrt(r1) * c
        return p

    def compute_surface_pcd(self, n_pts, min_dist=0.04):
        scene = self.scene()
        proba_faces = scene.area_faces
        proba_faces = proba_faces / proba_faces.sum()
        indices = np.random.choice(
            np.arange(scene.faces.shape[0]), p=proba_faces, size=n_pts
        )
        faces = scene.faces[indices]
        tris = scene.vertices[faces]
        points = self.sample_uniformly_triangle(tris)
        normals = scene.face_normals[indices]
        return points, normals

    def compute_volume_pcd(self, n_points):
        if len(self.geom_objs) == 0:
            return np.zeros((0, 3)), np.zeros((0, 3))

        # buggy function
        # points = trimesh.sample.volume_mesh(self.union_mesh, 6 * n_points)
        ray_intersector, scene = self.ray_intersector()
        rand_points = (
            np.random.random((6 * n_points, 3)) * scene.extents
        ) + scene.bounds[0]
        contains = ray_intersector.contains_points(rand_points)
        points = rand_points[contains]
        points, indices = utils.match_size(points, n_points)
        return points
