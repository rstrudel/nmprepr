import numpy as np

import eigenpy
import hppfcl
import pinocchio as pin


class Robot:
    def __init__(self):
        self.link_dim = None

    def get_neutral(self):
        return pin.neutral(self.model)

    def _set_collision_pairs(self, model, geom_model):
        raise NotImplementedError

    def _build_from_urdf(self, model_wrapper, urdf_path, package_path):
        model = model_wrapper.model
        geom_model = model_wrapper.geom_model

        pin.buildModelFromUrdf(urdf_path, model)
        pin.buildGeomFromUrdf(
            model, urdf_path, pin.GeometryType.COLLISION, geom_model, package_path,
        )
        # viz_model = pin.buildGeomFromUrdf(
        #     model, urdf_path, pin.GeometryType.VISUAL, [packages_path],
        # )

        # ! keep this line here for body dimensions to be defined (otherwize = -inf)
        model_data = model.createData()
        geom_model_data = geom_model.createData()
        # needed to get robot aabb, else returns inf value

    def _build_from_mesh(self, model_wrapper, mesh, bounds=None):
        model = model_wrapper.model
        geom_model = model_wrapper.geom_model

        # save geom_obj placement
        geom_obj = mesh.geom_obj()
        placement = geom_obj.placement.copy()
        geom_obj.placement = pin.SE3.Identity()
        geom_obj.q_placement = placement
        q_freeflyer = pin.SE3ToXYZQUAT(placement).copy()

        # create a freeflyer joint and add it to the model
        free_flyer = pin.JointModelFreeFlyer()
        universe_joint = 0
        joint_name = f"freeflyer_{mesh.name}"
        if bounds is not None:
            parent_joint = model.addJoint(
                universe_joint,
                free_flyer,
                pin.SE3.Identity(),
                joint_name=joint_name,
                max_effort=np.array([1000]),
                max_velocity=np.array([1000]),
                min_config=bounds[0],
                max_config=bounds[1],
            )
        else:
            parent_joint = model.addJoint(
                universe_joint, free_flyer, pin.SE3.Identity(), joint_name
            )

        # add geom_obj to the geom_model
        geom_obj.parentJoint = parent_joint
        geom_model.addGeometryObject(geom_obj)

    def project_q(self, q):
        return q

    def make_geom_obj(self):
        raise NotImplementedError
