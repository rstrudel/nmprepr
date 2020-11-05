import numpy as np
from mpenv.planning import rrt_bidir
from mpenv.planning import utils

EPSILON = 1e-7


def solve(env, delta_growth, iterations, simplify):
    """
    collision_fn : maps x to True (free) / False (collision)
    sample_fn : return a configuration
    """
    algo = rrt_bidir.rrt_bidir
    model_wrapper = env.model_wrapper
    delta_collision_check = env.delta_collision_check
    action_range = env.robot_props["action_range"]

    def collision_fn(q):
        return not model_wrapper.collision(q)

    def sample_fn():
        return model_wrapper.random_configuration()

    def distance_fn(q0, q1):
        return model_wrapper.distance(q0, q1)

    def interpolate_fn(q0, q1, t):
        return model_wrapper.interpolate(q0, q1, t)

    def arange_fn(q0, q1, resolution):
        return model_wrapper.arange(q0, q1, resolution)

    def expand_fn(q0, q1, limit_growth=True):
        if limit_growth:
            dist = distance_fn(q0, q1)
            t1 = min(dist, delta_growth) / (dist + EPSILON)
            q1 = interpolate_fn(q0, q1, t1)
        path = arange_fn(q0, q1, delta_collision_check)
        q_stop, collide = env.stopping_configuration(path)
        return q_stop, not collide

    def close_fn(qw0, qw1):
        return np.allclose(qw0.q, qw1.q)

    start = env.state
    goal = env.goal_state

    success, path, trees, iterations = algo(
        start, goal, sample_fn, expand_fn, distance_fn, close_fn, iterations=iterations
    )
    iterations_simplify = 0
    if success:
        if simplify:
            path["points"], iterations_simplify = utils.shorten(
                path["points"], expand_fn, interpolate_fn, distance_fn
            )
            path["points"] = utils.limit_step_size(
                path["points"], arange_fn, action_range
            )
        else:
            path["points"] = np.array(path["points"])
        path["collisions"] = np.array(path["collisions"])
        path["start"] = path["points"][0]
        path["goal"] = path["points"][-1]
    return success, path, trees, iterations + iterations_simplify
