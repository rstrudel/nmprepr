import numpy as np

import pinocchio as pin


class ModelWrapper:
    """
    Wrapper around Model, GeometryModel pinocchio classes,
    contains commonly used methods
    """

    def __init__(self, model=None, geom_model=None):
        if model is None:
            model = pin.Model()
        if geom_model is None:
            geom_model = pin.GeometryModel()
        self._model = model
        self._geom_model = geom_model
        self._data = None
        self._geom_data = None
        self._clip_bounds = (None, None)

    def configuration(self, q):
        return ConfigurationWrapper(self, q)

    @property
    def model(self):
        return self._model

    @property
    def geom_model(self):
        return self._geom_model

    @property
    def geom_data(self):
        return self._geom_data

    def create_data(self):
        self._data = self._model.createData()
        self._geom_data = self._geom_model.createData()

    def collision(self, qw):
        if isinstance(qw, ConfigurationWrapper):
            q = qw.q
        elif isinstance(qw, np.ndarray):
            qw = ConfigurationWrapper(self, qw)
            q = qw.q
        else:
            raise ValueError(
                "qw should either be a ConfigurationWrapper or a numpy array."
            )
        if q.shape[0] != self._model.nq:
            raise ValueError(
                f"The given configuration vector is of shape {q.shape[0]} while \
                the model requires a configuration vector of shape {self._model.nq}"
            )
        model = self._model
        data = self._data
        geom_model = self._geom_model
        geom_data = self._geom_data
        pin.forwardKinematics(model, data, q)
        pin.updateGeometryPlacements(model, data, geom_model, geom_data)
        qw.oMi = data.oMi.tolist()
        qw.oMg = geom_data.oMg.tolist()
        # stop at the first collision
        collide = pin.computeCollisions(geom_model, geom_data, True)
        return collide

    def collision_pairs(self):
        cps = self.geom_model.collisionPairs
        crs = self.geom_data.collisionResults
        pairs = []
        results = []
        for cp, cr in zip(cps, crs):
            pairs.append((cp.first, cp.second))
            results.append(cr.isCollision())
        return pairs, results

    def collision_labels(self):
        pairs, results = self.collision_pairs()
        n_geoms = len(self.geom_model.geometryObjects)
        collision_labels = np.zeros(n_geoms, dtype=bool)
        for pair, result in zip(pairs, results):
            if result:
                collision_labels[pair[0]] = True
                collision_labels[pair[1]] = True
        return collision_labels

    def distance(self, qw0, qw1):
        q0, q1 = qw0.q, qw1.q
        return pin.distance(self._model, q0, q1)

    def arange(self, qw0, qw1, delta):
        """
        return points from q0 to q1 evenly spaced with distance delta
        """
        model = self._model
        d = self.distance(qw0, qw1)
        # ensure at least one point is generated
        d = max(d, delta)
        n_pts = np.ceil(d / delta).astype(int)
        steps = np.linspace(0, 1, num=n_pts, endpoint=True)
        path = []
        for i, t in enumerate(steps):
            path.append(self.interpolate(qw0, qw1, t))
        return path

    def interpolate(self, qw0, qw1, t):
        q0, q1 = qw0.q, qw1.q
        q = pin.interpolate(self._model, q0, q1, t)
        qw = ConfigurationWrapper(self, q)
        return qw

    def integrate(self, qw, v, cartesian):
        q0 = qw.q
        if cartesian:
            q1 = qw.q + v
        else:
            q1 = pin.integrate(self._model, q0, v)
        qw1 = ConfigurationWrapper(self, q1)
        return qw1

    def neutral_configuration(self):
        q = pin.neutral(self._model)
        qw = ConfigurationWrapper(self, q)
        return qw

    def random_configuration(self, project_fn=None):
        q = pin.randomConfiguration(self.model)
        # q = np.random.uniform(-np.pi, np.pi, size=(6,))
        qw = ConfigurationWrapper(self, q)
        qw = self.clip(qw, self._clip_bounds[0], self._clip_bounds[1])[1]
        if project_fn:
            qw = project_fn(qw)
        return qw

    def random_free_configuration(self, project_fn=None):
        collide = True
        while collide:
            qw = self.random_configuration(project_fn)
            collide = self.collision(qw)
        return qw

    def set_clipping(self, min_q, max_q):
        self._clip_bounds = [min_q, max_q]

    def clip(self, qw, min_q, max_q):
        q = qw.q
        clip_bounds = np.zeros((2, q.shape[0]))
        clip_bounds[0, :] = -np.inf
        clip_bounds[1, :] = np.inf
        if min_q is not None:
            n_clip_bounds = min_q.shape[0]
            clip_bounds[0, :n_clip_bounds] = min_q
        if max_q is not None:
            n_clip_bounds = max_q.shape[0]
            clip_bounds[1, :n_clip_bounds] = max_q
        clipped_q = np.clip(q, clip_bounds[0], clip_bounds[1])
        clipped = not np.allclose(q, clipped_q)
        clipped_qw = ConfigurationWrapper(self, clipped_q)
        return clipped, clipped_qw

    def compute_oMi(self, qw):
        q = qw.q
        model = self._model
        data = self._data
        pin.forwardKinematics(model, data, q)
        qw.oMi = data.oMi.tolist()

    def compute_oMg(self, qw):
        q = qw.q
        model = self._model
        data = self._data
        geom_model = self._geom_model
        geom_data = self._geom_data
        pin.forwardKinematics(model, data, q)
        pin.updateGeometryPlacements(model, data, geom_model, geom_data, q)
        qw.oMg = geom_data.oMg.tolist()

    def copy(self):
        return self.__copy__()

    def __copy__(self):
        return ModelWrapper(self._model.copy(), self._geom_model.copy())


class ConfigurationWrapper:
    """
    Wrapper to avoid repeated computations associated to a configuration,
    update only the position of joints and geometries when needed
    """

    def __init__(self, model_wrapper, q):
        # updated is True if oMi/oMg corresponds to the current q
        self._model_wrapper = model_wrapper
        if not isinstance(q, np.ndarray):
            raise ValueError("q should be a numpy array.")
        self.q = q

    def __repr__(self):
        return "ConfigurationWrapper(" + np.array_str(self._q) + ")"

    @property
    def q(self):
        return self._q.copy()

    @property
    def oMi(self):
        if self._oMi is None:
            self._model_wrapper.compute_oMi(self)
        return self._oMi

    @property
    def oMg(self):
        if self._oMg is None:
            self._model_wrapper.compute_oMg(self)
        return self._oMg

    @property
    def q_oM(self):
        return self.q, self.oMi, self.oMg

    @q.setter
    def q(self, q):
        self._q = q
        self._oMi = None
        self._oMg = None

    @oMi.setter
    def oMi(self, oMi):
        self._oMi = oMi

    @oMg.setter
    def oMg(self, oMg):
        self._oMg = oMg
