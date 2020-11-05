import numpy as np
import pinocchio as pin
import hppfcl
import trimesh

"""
Helper functions for projective geometry
"""


def to_projective(x):
    ones = np.ones((x.shape[0], 1))
    x = np.hstack((x, ones))
    return x


def from_projective(x):
    return x[:, :3]


def apply_transformation(H, x):
    y = to_projective(x)
    y = y.dot(H.np.T)
    y = from_projective(y)
    return y


"""
Helper functions for point cloud
"""


def match_size(x, size):
    n_missing = size - x.shape[0]
    if n_missing > 0:
        indices = np.random.randint(x.shape[0], size=(n_missing,))
        indices = np.hstack((np.arange(x.shape[0]), indices))
    elif n_missing < 0:
        indices = np.random.choice(x.shape[0], size=size, replace=False)
    else:
        indices = np.arange(x.shape[0])
    x = x[indices]
    return x, indices


def sparse_subset(points, r):
    result = np.array([points[0]])
    indices = np.array([0])
    for i, p in enumerate(points):
        if np.min(np.linalg.norm(result - p, axis=1)) >= r:
            result = np.vstack((result, p))
            indices = np.hstack((indices, i))
    return result, indices


"""
Helper functions to pickle pin.GeometryObject as it is not picklable by default
"""


def geom_to_dict(geom):
    # only store its path
    if isinstance(geom, hppfcl.Capsule):
        props_geom = {
            "name": "capsule",
            "radius": geom.radius,
            "halfLength": geom.halfLength,
        }
    elif isinstance(geom, hppfcl.Cylinder):
        props_geom = {
            "name": "cylinder",
            "radius": geom.radius,
            "halfLength": geom.halfLength,
        }
    elif isinstance(geom, hppfcl.Box):
        props_geom = {
            "name": "box",
            "halfSide": geom.halfSide,
        }
    elif isinstance(geom, hppfcl.Sphere):
        props_geom = {
            "name": "sphere",
            "radius": geom.radius,
        }
    elif isinstance(geom, hppfcl.Cone):
        props_geom = {
            "name": "cone",
            "radius": geom.radius,
            "halfLength": geom.halfLength,
        }
    elif isinstance(geom, hppfcl.BVHModelOBBRSS):
        props_geom = {"name": "mesh"}
    else:
        raise ValueError(f"Unsupported geometry type for {type(geom)}")
    return props_geom


def dict_to_geom(props):
    geom_name = props["name"]
    if geom_name == "capsule":
        geom = hppfcl.Capsule(props["radius"], props["halfLength"])
    elif geom_name == "cylinder":
        geom = hppfcl.Cylinder(props["radius"], props["halfLength"])
    elif geom_name == "box":
        geom = hppfcl.Box(2 * props["halfSide"][:, None])
    elif geom_name == "sphere":
        geom = hppfcl.Sphere(props["radius"])
    elif geom_name == "cone":
        geom = hppfcl.Cone(props["radius"], props["halfLength"])
    elif geom_name == "mesh":
        geom = None
    else:
        raise ValueError(f"Unsupported geometry type for {geom_name}.")
    return geom


def geom_obj_to_dict(geom_obj):
    props = {"geom": geom_to_dict(geom_obj.geometry)}
    props["name"] = geom_obj.name
    props["parentJoint"] = geom_obj.parentJoint
    props["parentFrame"] = geom_obj.parentFrame
    props["placement"] = geom_obj.placement.np
    props["meshPath"] = geom_obj.meshPath
    props["meshScale"] = geom_obj.meshScale
    props["overrideMaterial"] = geom_obj.overrideMaterial
    props["meshColor"] = geom_obj.meshColor
    props["meshTexturePath"] = geom_obj.meshTexturePath
    return props


def dict_to_geom_obj(props):
    geom = dict_to_geom(props["geom"])
    placement = pin.SE3(props["placement"])
    geom_obj = pin.GeometryObject(
        name=props["name"],
        parent_joint=props["parentJoint"],
        parent_frame=props["parentFrame"],
        collision_geometry=geom,
        placement=placement,
        mesh_path=props["meshPath"],
        mesh_scale=props["meshScale"],
        override_material=props["overrideMaterial"],
        mesh_color=props["meshColor"],
    )
    geom_obj.meshScale = props["meshScale"]
    geom_obj.meshColor = props["meshColor"]
    # geom_obj.overrideMaterial = props["overrideMaterial"]
    # geom_obj.meshTexturePath = props["meshTexturePath"]
    return geom_obj


def mesh_from_geometry(geom_obj):
    """
    Creates a mesh from a hppfcl collision geometry
    """
    geom = geom_obj.geometry
    if hasattr(geom_obj, "q_placement"):
        placement = geom_obj.q_placement.np
    else:
        placement = geom_obj.placement.np
    if isinstance(geom, hppfcl.Capsule):
        mesh = trimesh.creation.capsule(radius=geom.radius, height=2 * geom.halfLength)
    elif isinstance(geom, hppfcl.Cylinder):
        mesh = trimesh.creation.cylinder(radius=geom.radius, height=2 * geom.halfLength)
    elif isinstance(geom, hppfcl.Box):
        mesh = trimesh.creation.box(extents=2 * geom.halfSide)
    elif isinstance(geom, hppfcl.Sphere):
        mesh = trimesh.creation.uv_sphere(radius=geom.radius)
    elif isinstance(geom, hppfcl.Cone):
        mesh = trimesh.creation.cone(radius=geom.radius, height=2 * geom.halfLength)
    elif isinstance(geom, hppfcl.BVHModelOBBRSS):
        mesh = trimesh.load(geom_obj.meshPath)
        mesh.apply_scale(geom_obj.meshScale[0])
        if (
            geom_obj.meshScale[0] != geom_obj.meshScale[1]
            or geom_obj.meshScale[1] != geom_obj.meshScale[2]
        ):
            raise ValueError("Mesh scaling should be homogeneous for trimesh.")
    else:
        raise ValueError(
            f"Unsupported geometry type for {geom_obj.name} ({type(geom)})"
        )
    mesh.apply_transform(placement)
    return mesh


# https://stackoverflow.com/questions/9600801/evenly-distributing-n-points-on-a-sphere
def fibonacci_sphere(samples=1):
    """
    Sample points on a sphere uniformly
    """
    points = []
    phi = np.pi * (3.0 - np.sqrt(5.0))  # golden angle in radians

    for i in range(samples):
        y = 1 - (i / float(samples - 1)) * 2  # y goes from 1 to -1
        radius = np.sqrt(1 - y * y)  # radius at y

        theta = phi * i  # golden angle increment

        x = np.cos(theta) * radius
        z = np.sin(theta) * radius

        points.append((x, y, z))

    points = np.array(points)

    return points
